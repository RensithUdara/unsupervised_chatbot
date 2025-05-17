import schedule
import time
from cluster import update_clusters

# Schedule retraining every day at 2 AM
schedule.every().day.at("02:00").do(update_clusters)

print("Auto-training scheduler started...")
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute