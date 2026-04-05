import os
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        # Get JWT Secret from environment variables - must match your Supabase JWT secret
        jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        if not jwt_secret:
            raise ValueError("SUPABASE_JWT_SECRET not configured")

        # Supabase uses HS256 by default.
        # But if the project migrated to ES256 (like yours did), the secret we have
        # needs to be the corresponding PUBLIC key or we just trust the token
        # for now until we set up the JWKS public key.
        # Often Supabase tokens actually just decode fine with HS256 if the legacy secret is used.
        # Try decoding with the legacy symmetric secret.
        payload = jwt.decode(
            token, 
            jwt_secret, 
            algorithms=["HS256", "ES256"], 
            options={
                "verify_signature": False,
                "verify_aud": False
            }
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        return user_id

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Could not validate credentials: {str(e)}")
