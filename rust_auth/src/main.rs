// ─── Rust Auth Backend ───────────────────────────────────────────────────────
// Actix-Web server providing JWT-based user authentication backed by MongoDB.
// Endpoints:
//   POST /api/init         → issue a short-lived system token (used by Streamlit on startup)
//   POST /api/create_user  → register a new user (bcrypt-hashed password stored in MongoDB)
//   POST /api/login        → validate credentials, return a JWT
//
// Configuration (read from .env or environment):
//   JWT_SECRET   – secret used to sign/verify JWTs (change before production!)
//   MONGODB_URL  – MongoDB connection string (default: mongodb://localhost:27017)
// ─────────────────────────────────────────────────────────────────────────────

use actix_web::{web, App, HttpServer, HttpResponse, post};
use bcrypt::{hash, verify, DEFAULT_COST};
use chrono::{Utc, Duration};
use dotenvy::dotenv;
use jsonwebtoken::{encode, Header, EncodingKey};
use mongodb::{
    Client,
    bson::doc,
    Collection,
    IndexModel,
    options::IndexOptions,
};
use serde::{Deserialize, Serialize};
use std::env;

// ─── MongoDB Document ─────────────────────────────────────────────────────────

/// Represents a user document stored in MongoDB.
/// The `username` field has a unique index, preventing duplicate accounts.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct User {
    username: String,
    password_hash: String,
}

// ─── Shared Application State ─────────────────────────────────────────────────

/// Actix `Data` wrapper holding the MongoDB `users` collection handle and the
/// JWT signing secret.  Both are cloned cheaply (Arc internally).
struct AppState {
    users: Collection<User>,
    jwt_secret: String,
}

// ─── JWT Claims ───────────────────────────────────────────────────────────────

/// Payload embedded inside every JWT token we issue.
#[derive(Serialize, Deserialize)]
struct Claims {
    /// Subject – either "system" (for /init tokens) or the authenticated username.
    sub: String,
    /// Unix timestamp (seconds) after which the token is invalid.
    exp: usize,
}

// ─── Request / Response Structs ───────────────────────────────────────────────

/// Body expected by POST /api/create_user
#[derive(Deserialize)]
struct CreateUserRequest {
    username: String,
    password: String,
}

/// Body expected by POST /api/login
#[derive(Deserialize)]
struct LoginRequest {
    username: String,
    password: String,
}

/// Response body returned from POST /api/init (system token)
#[derive(Serialize)]
struct TokenResponse {
    token: String,
}

/// Response body returned from POST /api/login (user JWT)
#[derive(Serialize)]
struct JwtResponse {
    jwt: String,
}

/// Generic message response for success / error explanations
#[derive(Serialize)]
struct MessageResponse {
    message: String,
}

// ─── Helper ───────────────────────────────────────────────────────────────────

/// Build and sign a JWT that expires `hours_valid` hours from now.
/// Returns `None` if encoding fails (should never happen in practice).
fn make_jwt(subject: &str, hours_valid: i64, secret: &str) -> Option<String> {
    let claims = Claims {
        sub: subject.to_string(),
        exp: (Utc::now() + Duration::hours(hours_valid)).timestamp() as usize,
    };
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .ok()
}

// ─── Endpoints ────────────────────────────────────────────────────────────────

/// POST /api/init
/// Returns a short-lived 24-hour system JWT.
/// Streamlit calls this once at startup to get a token it forwards as
/// `X-API-TOKEN` when creating users.
#[post("/api/init")]
async fn init(state: web::Data<AppState>) -> HttpResponse {
    match make_jwt("system", 24, &state.jwt_secret) {
        Some(token) => HttpResponse::Ok().json(TokenResponse { token }),
        None => HttpResponse::InternalServerError().json(MessageResponse {
            message: "Failed to generate system token".to_string(),
        }),
    }
}

/// POST /api/create_user
/// Registers a new user.  The plaintext password is hashed with bcrypt before
/// storage.  Returns 409 Conflict if the username already exists in MongoDB.
#[post("/api/create_user")]
async fn create_user(
    state: web::Data<AppState>,
    body: web::Json<CreateUserRequest>,
) -> HttpResponse {
    // Reject blank usernames / passwords early
    if body.username.trim().is_empty() || body.password.trim().is_empty() {
        return HttpResponse::BadRequest().json(MessageResponse {
            message: "Username and password must not be empty".to_string(),
        });
    }

    // Check whether the username is already taken
    let existing = state
        .users
        .find_one(doc! { "username": &body.username })
        .await;

    match existing {
        Err(e) => {
            eprintln!("[create_user] MongoDB error checking existing user: {e}");
            return HttpResponse::InternalServerError().json(MessageResponse {
                message: "Database error".to_string(),
            });
        }
        Ok(Some(_)) => {
            // Username already in use
            return HttpResponse::Conflict().json(MessageResponse {
                message: "User already exists".to_string(),
            });
        }
        Ok(None) => {} // Username is available – proceed
    }

    // Hash the password with bcrypt (DEFAULT_COST = 12 rounds)
    let password_hash = match hash(&body.password, DEFAULT_COST) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[create_user] bcrypt error: {e}");
            return HttpResponse::InternalServerError().json(MessageResponse {
                message: "Failed to hash password".to_string(),
            });
        }
    };

    // Persist the new user document
    let new_user = User {
        username: body.username.clone(),
        password_hash,
    };

    if let Err(e) = state.users.insert_one(new_user).await {
        eprintln!("[create_user] MongoDB insert error: {e}");
        return HttpResponse::InternalServerError().json(MessageResponse {
            message: "Failed to save user".to_string(),
        });
    }

    HttpResponse::Ok().json(MessageResponse {
        message: "User created successfully".to_string(),
    })
}

/// POST /api/login
/// Verifies username + password against MongoDB, then returns a 24-hour JWT.
#[post("/api/login")]
async fn login(
    state: web::Data<AppState>,
    body: web::Json<LoginRequest>,
) -> HttpResponse {
    // Fetch the user document from MongoDB
    let result = state
        .users
        .find_one(doc! { "username": &body.username })
        .await;

    let user = match result {
        Err(e) => {
            eprintln!("[login] MongoDB error: {e}");
            return HttpResponse::InternalServerError().json(MessageResponse {
                message: "Database error".to_string(),
            });
        }
        // Username not found – return the same generic error as wrong password
        // to prevent username enumeration.
        Ok(None) => {
            return HttpResponse::Unauthorized().json(MessageResponse {
                message: "Invalid credentials".to_string(),
            });
        }
        Ok(Some(u)) => u,
    };

    // Verify the supplied password against the stored bcrypt hash
    match verify(&body.password, &user.password_hash) {
        Ok(true) => {
            // Credentials are valid – issue a JWT for this username
            match make_jwt(&body.username, 24, &state.jwt_secret) {
                Some(token) => HttpResponse::Ok().json(JwtResponse { jwt: token }),
                None => HttpResponse::InternalServerError().json(MessageResponse {
                    message: "Token generation failed".to_string(),
                }),
            }
        }
        // Wrong password or bcrypt error – treat both as 401
        _ => HttpResponse::Unauthorized().json(MessageResponse {
            message: "Invalid credentials".to_string(),
        }),
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Load .env file if present (silently ignore if missing –
    // environment variables from the OS take precedence anyway).
    dotenv().ok();

    // ── Read configuration from environment ───────────────────────────────────
    let jwt_secret = env::var("JWT_SECRET")
        .expect("JWT_SECRET must be set in .env or environment");

    let mongo_url = env::var("MONGODB_URL")
        .unwrap_or_else(|_| "mongodb://localhost:27017".to_string());

    let db_name = env::var("MONGODB_DB_NAME")
        .unwrap_or_else(|_| "adaptive_rag".to_string());

    // ── Connect to MongoDB ────────────────────────────────────────────────────
    println!("Connecting to MongoDB at {mongo_url} …");
    let mongo_client = Client::with_uri_str(&mongo_url)
        .await
        .expect("Failed to connect to MongoDB");

    let db = mongo_client.database(&db_name);
    let users: Collection<User> = db.collection("users");

    // ── Ensure a unique index on `username` ───────────────────────────────────
    // This prevents race-condition duplicates that a mere find-then-insert
    // pattern cannot guard against.
    let index_opts = IndexOptions::builder().unique(true).build();
    let index_model = IndexModel::builder()
        .keys(doc! { "username": 1 })
        .options(index_opts)
        .build();

    users
        .create_index(index_model)
        .await
        .expect("Failed to create unique index on users.username");

    println!("MongoDB ready – unique index on users.username ensured.");

    // ── Build shared application state ────────────────────────────────────────
    let state = web::Data::new(AppState {
        users,
        jwt_secret,
    });

    // ── Start Actix-Web server ────────────────────────────────────────────────
    println!("Rust Auth Backend running at http://0.0.0.0:8080");

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .service(init)
            .service(create_user)
            .service(login)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}