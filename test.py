from datetime import datetime

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

current_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')

#current_time = datetime.now()

print(current_time.replace(hour=0, minute=0, second=0, microsecond=0))