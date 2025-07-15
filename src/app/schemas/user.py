from pydantic import BaseModel

class UserRegister(BaseModel):
    username: str
    password: str
    secret_key: str

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
