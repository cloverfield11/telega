import os.path
import pathlib
import json
from datetime import date

def year_month(date):
    return str(date)[:7]

class UsageTracker:
    def __init__(self, user_id, user_name, logs_dir="usage_logs"):
        self.user_id = user_id
        self.logs_dir = logs_dir
        self.user_file = f"{logs_dir}/{user_id}.json"
        if os.path.isfile(self.user_file):
            with open(self.user_file, "r") as file:
                self.usage = json.load(file)
        else:
            pathlib.Path(logs_dir).mkdir(exist_ok=True)
            self.usage = {
                "user_name": user_name,
                "current_cost": {"day": 0.0, "month": 0.0, "all_time": 0.0, "last_update": str(date.today())},
                "usage_history": {"chat_tokens": {}, "transcription_seconds": {}, "number_images": {}}
            }

    def add_image_request(self, image_size, image_prices="0.016,0.018,0.02"):
        sizes = ["256x256", "512x512", "1024x1024"]
        requested_size = sizes.index(image_size)
        image_cost = image_prices[requested_size]
        today = date.today()
        last_update = date.fromisoformat(self.usage["current_cost"]["last_update"])
        self.usage["current_cost"]["all_time"] = self.usage["current_cost"].get("all_time", self.initialize_all_time_cost()) + image_cost
        if today == last_update:
            self.usage["current_cost"]["day"] += image_cost
            self.usage["current_cost"]["month"] += image_cost
        else:
            if today.month == last_update.month:
                self.usage["current_cost"]["month"] += image_cost
            else:
                self.usage["current_cost"]["month"] = image_cost
            self.usage["current_cost"]["day"] = image_cost
            self.usage["current_cost"]["last_update"] = str(today)
        if str(today) in self.usage["usage_history"]["number_images"]:
            self.usage["usage_history"]["number_images"][str(today)][requested_size] += 1
        else:
            self.usage["usage_history"]["number_images"][str(today)] = [0, 0, 0]
            self.usage["usage_history"]["number_images"][str(today)][requested_size] += 1
        with open(self.user_file, "w") as outfile:
            json.dump(self.usage, outfile)

    def get_current_image_count(self):
        today=date.today()
        if str(today) in self.usage["usage_history"]["number_images"]:
            usage_day = sum(self.usage["usage_history"]["number_images"][str(today)])
        else:
            usage_day = 0
        month = str(today)[:7] # year-month as string
        usage_month = 0
        for today, images in self.usage["usage_history"]["number_images"].items():
            if today.startswith(month):
                usage_month += sum(images)
        return usage_day, usage_month