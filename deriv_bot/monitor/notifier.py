"""
Module for sending notifications about important trading events
"""
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class Notifier:
    def __init__(self, email_config=None):
        """
        Initialize notifier

        Args:
            email_config: Dictionary containing email configuration
        """
        self.email_config = email_config or {}
        self.notification_queue = []

    def add_notification(self, level, message):
        """
        Add notification to queue

        Args:
            level: Notification level ('info', 'warning', 'error')
            message: Notification message
        """
        try:
            notification = {
                'level': level,
                'message': message,
                'timestamp': datetime.now()
            }

            self.notification_queue.append(notification)
            logger.info(f"Notification added: {message}")

            # Process notifications if queue gets too long
            if len(self.notification_queue) >= 10:
                self.process_notifications()

        except Exception as e:
            logger.error(f"Error adding notification: {str(e)}")

    def process_notifications(self):
        """Process and send pending notifications"""
        try:
            if not self.notification_queue:
                return

            # Group notifications by level
            grouped = {}
            for notification in self.notification_queue:
                level = notification['level']
                if level not in grouped:
                    grouped[level] = []
                grouped[level].append(notification)

            # Send email notifications
            if self.email_config:
                self._send_email_notifications(grouped)

            # Clear queue
            self.notification_queue = []

        except Exception as e:
            logger.error(f"Error processing notifications: {str(e)}")

    def _send_email_notifications(self, grouped_notifications):
        """Send email notifications"""
        try:
            if not self.email_config:
                return

            # Prepare email content
            content = []
            for level, notifications in grouped_notifications.items():
                content.append(f"\n{level.upper()} Notifications:")
                for notification in notifications:
                    content.append(
                        f"[{notification['timestamp']}] {notification['message']}"
                    )

            email_body = "\n".join(content)

            # Create email message
            msg = MIMEText(email_body)
            msg['Subject'] = 'Trading Bot Notifications'
            msg['From'] = self.email_config.get('from_email')
            msg['To'] = self.email_config.get('to_email')

            # Send email
            with smtplib.SMTP(
                self.email_config.get('smtp_server', 'localhost'),
                self.email_config.get('smtp_port', 25)
            ) as server:
                if self.email_config.get('use_tls'):
                    server.starttls()

                if 'username' in self.email_config:
                    server.login(
                        self.email_config['username'],
                        self.email_config['password']
                    )

                server.send_message(msg)

            logger.info("Email notifications sent")

        except Exception as e:
            logger.error(f"Error sending email notifications: {str(e)}")

    def notify_trade_executed(self, trade_details):
        """Send notification for executed trade"""
        message = (
            f"Trade executed: {trade_details['symbol']} "
            f"{trade_details['contract_type']} "
            f"Amount: {trade_details['amount']}"
        )
        self.add_notification('info', message)

    def notify_risk_warning(self, warning_message):
        """Send risk-related warning notification"""
        self.add_notification('warning', f"Risk Warning: {warning_message}")

    def notify_error(self, error_message):
        """Send error notification"""
        self.add_notification('error', f"Error: {error_message}")