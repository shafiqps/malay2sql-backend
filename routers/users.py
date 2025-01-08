from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from database import get_db
from models.user import UserDB, UserUpdate, UserResponse
from routers.auth import oauth2_scheme, jwt, SECRET_KEY, ALGORITHM
from jose import JWTError 
from services.storage_service import CloudStorageService
from typing import Optional
import imghdr

router = APIRouter(prefix="/users", tags=["users"])
# Initialize storage service
storage_service = CloudStorageService(bucket_name="malay2sql-bucket")

def validate_image(file: UploadFile) -> bool:
    header = file.file.read(512)
    file.file.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return False
    allowed_formats = ['jpeg', 'png', 'gif']
    return format.lower() in allowed_formats


    
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(UserDB).filter(UserDB.email == email).first()
    if user is None:
        raise credentials_exception
    return user

@router.post("/me/profile-picture")
async def upload_profile_picture(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user)
):
    """Upload a profile picture"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not validate_image(file):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    try:
        # Delete old profile picture if exists
        if current_user.profile_picture_url:
            await storage_service.delete_profile_picture(current_user.profile_picture_url)
        
        # Upload new profile picture
        url = await storage_service.upload_profile_picture(current_user.id, file)
        
        # Update user profile
        current_user.profile_picture_url = url
        db.commit()
        
        return {"message": "Profile picture updated successfully", "url": url}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/me/profile-picture")
async def delete_profile_picture(
    db: Session = Depends(get_db),
    current_user: UserDB = Depends(get_current_user)
):
    """Delete user's profile picture"""
    if not current_user.profile_picture_url:
        raise HTTPException(status_code=404, detail="No profile picture found")
    
    try:
        # Delete from Cloud Storage
        await storage_service.delete_profile_picture(current_user.profile_picture_url)
        
        # Update user profile
        current_user.profile_picture_url = None
        db.commit()
        
        return {"message": "Profile picture deleted successfully"}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserDB = Depends(get_current_user)):
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_user(user_update: UserUpdate, current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    for key, value in user_update.dict(exclude_unset=True).items():
        setattr(current_user, key, value)
    db.commit()
    db.refresh(current_user)
    return current_user