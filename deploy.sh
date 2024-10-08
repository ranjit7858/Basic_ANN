#!/bin/bash

echo "deleting old app"
sudo rm -rf /var/www/

echo "creating app folder"
sudo mkdir -p /var/www/Customer_Churn_Prediction_ANN

echo "moving files to app folder"
sudo mv  * /var/www/Customer_Churn_Prediction_ANN


# python -m venv venv
# source venv/bin/activate

# sudo apt update
# sudo apt install python3-uwsgi python3-gevent

# Navigate to the app directory
cd /var/www/Customer_Churn_Prediction_ANN/
# sudo mv env .env



sudo apt-get update -y
echo "installing python and pip"
sudo apt-get install -y python3 python3-pip

#launch venv
python3 -m venv venv
source venv/bin/activate

# added few dependencies from taipy doc.
sudo apt update -y
sudo apt install -y python3-pip nginx
sudo pip install uwsgi gevent 
# sudo ln -s `pwd`/.local/bin/uwsgi /usr/bin/uwsgi

# Install application dependencies from requirements.txt
echo "Install application dependencies from requirements.txt"
sudo pip install -r requirements.txt --no-cache-dir

# Update and install Nginx if not already installed
if ! command -v nginx > /dev/null; then
    echo "Installing Nginx"
    sudo apt-get update -y
    sudo apt-get install -y nginx
fi

# Configure Nginx to act as a reverse proxy if not already configured
if [ ! -f /etc/nginx/sites-available/myapp ]; then
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo bash -c 'cat > /etc/nginx/sites-available/myapp <<EOF
server {
    listen 80;
    server_name _;

    location / {
        include proxy_params;
        proxy_pass http://unix:/var/www/Customer_Churn_Prediction_ANN/myapp.sock;
    }
}
EOF'

    sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled
    sudo systemctl restart nginx
else
    echo "Nginx reverse proxy configuration already exists."
fi


# sudo pipx install uwsgi gevent
echo "starting uwsgi ................................................."
uwsgi --http 0.0.0.0:5000 --gevent 1000 --http-websockets --module app:web_app
echo "started uwsgi 🚀................................................"
# Stop any existing Gunicorn process
# sudo pkill gunicorn``
# # sudo pkill uwsgi
# sudo rm -rf myapp.sock

# # # Start Gunicorn with the Flask application
# # # Replace 'server:app' with 'yourfile:app' if your Flask instance is named differently.
# # gunicorn --workers 3 --bind 0.0.0.0:8000 server:app &
# echo "starting gunicorn"
# sudo gunicorn --workers 3 --bind unix:myapp.sock  app:app --user www-data --group www-data --daemon
# echo "started gunicorn 🚀"