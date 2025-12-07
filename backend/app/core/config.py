from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Trading Bot API"
    API_V1_STR: str = "/api"
    
    # Add other settings here (database, keys, etc.)
    
    class Config:
        case_sensitive = True

settings = Settings()
