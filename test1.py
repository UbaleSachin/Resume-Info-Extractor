"""Certainly! Here’s a complete flow for a login system with a frosted glass UI, JWT authentication, and MySQL user data storage.
This includes:

Frontend login page (frosted glass style)
Backend FastAPI endpoints for register/login
JWT token handling
MySQL user table and async DB access
Example: how to save/retrieve user data (e.g., resumes) per user

pip install fastapi sqlalchemy aiomysql passlib[bcrypt] python-jose[cryptography] python-dotenv

2. Backend: FastAPI + MySQL + JWT
Add to your backend (e.g., main.py):

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import os, asyncio

DATABASE_URL = "mysql+aiomysql://username:password@localhost:3306/yourdbname"
SECRET_KEY = "YOUR_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True)
    hashed_password = Column(String(128))

async def create_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

class UserIn(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

app = FastAPI()

@app.on_event("startup")
async def startup():
    await create_db()

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_user(session, username: str):
    result = await session.execute(
        User.__table__.select().where(User.username == username)
    )
    return result.scalar_one_or_none()

def create_access_token(data: dict):
    from datetime import datetime, timedelta
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    async with AsyncSessionLocal() as session:
        user = await get_user(session, username)
        if user is None:
            raise credentials_exception
        return user

@app.post("/register")
async def register(user: UserIn):
    async with AsyncSessionLocal() as session:
        db_user = await get_user(session, user.username)
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        hashed_pw = get_password_hash(user.password)
        new_user = User(username=user.username, hashed_password=hashed_pw)
        session.add(new_user)
        await session.commit()
        return {"msg": "User registered"}

@app.post("/login", response_model=Token)
async def login(form_data: UserIn):
    async with AsyncSessionLocal() as session:
        user = await get_user(session, form_data.username)
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        access_token = create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer"}

# Example: Save extracted data for a user
@app.post("/save-extracted-data")
async def save_extracted_data(data: dict, user=Depends(get_current_user)):
    # Save data to a table with user_id reference (implement your own model)
    # Example: ResumeData(user_id=user.id, data=json.dumps(data))
    return {"msg": "Data saved for user", "user": user.username}

# Example: Get extracted data for a user
@app.get("/get-extracted-data")
async def get_extracted_data(user=Depends(get_current_user)):
    # Query your ResumeData table for user_id=user.id
    return {"msg": "Fetched data for user", "user": user.username}


3. Frontend: Use JWT for API Calls
In your main app JS, send the JWT token with each API request:

const token = localStorage.getItem('token');
const resp = await fetch('http://localhost:8000/save-extracted-data', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ ...yourData })
});


4. How the Flow Works
User registers or logs in at /login.html.
JWT token is saved in localStorage.
All API requests include the JWT in the Authorization header.
Backend authenticates the user and saves/loads data from MySQL, associated with the user.
User can log in from any device and see their saved data.


5. Summary Table
Step	Frontend (frosted glass)	Backend (FastAPI + MySQL)	Auth
Register	/register	/register endpoint	No
Login	/login	/login endpoint	No
Save Data	/save-extracted-data	/save-extracted-data	Yes
Get Data	/get-extracted-data	/get-extracted-data	Yes
You now have a secure, persistent, multi-user system with a modern frosted glass UI!

Let me know if you want a full MySQL table schema for storing resumes or more advanced user features.

Similar code found with 2 license types - View matches"""