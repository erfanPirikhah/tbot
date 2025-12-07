# Simple config without BaseSettings to avoid dependency issues
class Settings:
    PROJECT_NAME: str = "Trading Bot API"
    API_V1_STR: str = "/api"
    
    def __init__(self):
        self.PROJECT_NAME = "Trading Bot API"
        self.API_V1_STR = "/api"

settings = Settings()
