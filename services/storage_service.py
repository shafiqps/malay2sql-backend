from google.cloud import storage
from fastapi import UploadFile
import uuid
from PIL import Image
from io import BytesIO
import logging

class CloudStorageService:
    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)

    async def upload_profile_picture(self, user_id: int, file: UploadFile) -> str:
        try:
            # Read and process image
            contents = await file.read()
            image = Image.open(BytesIO(contents))
            
            # Resize image
            image.thumbnail((500, 500))
            
            # Convert to bytes
            output = BytesIO()
            image.save(output, format='PNG')
            output.seek(0)
            
            # Generate unique filename
            filename = f"profile_pictures/{user_id}/{str(uuid.uuid4())}.png"
            
            # Upload to Cloud Storage
            blob = self.bucket.blob(filename)
            blob.upload_from_file(
                output, 
                content_type='image/png',
            )
            
            # Generate public URL
            return f"https://storage.googleapis.com/{self.bucket_name}/{filename}"
            
        except Exception as e:
            logging.error(f"Failed to upload profile picture: {str(e)}")
            raise

    async def delete_profile_picture(self, url: str) -> None:
        try:
            # Extract filename from URL
            filename = url.split(f"storage.googleapis.com/{self.bucket_name}/")[1]
            
            # Delete from Cloud Storage
            blob = self.bucket.blob(filename)
            blob.delete()
            
        except Exception as e:
            logging.error(f"Failed to delete profile picture: {str(e)}")
            raise