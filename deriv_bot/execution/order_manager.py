"""
Module for managing and tracking trading orders
"""
from datetime import datetime
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class OrderManager:
    def __init__(self):
        self.active_orders = {}
        self.order_history = []
        
    async def place_order(self, order_executor, order_params):
        """
        Place and track a new order
        
        Args:
            order_executor: OrderExecutor instance
            order_params: Dictionary containing order parameters
        """
        try:
            # Execute order
            order_result = await order_executor.place_order(
                order_params['symbol'],
                order_params['contract_type'],
                order_params['amount'],
                order_params['duration']
            )
            
            if order_result:
                # Track order
                order_id = order_result['contract_id']
                self.active_orders[order_id] = {
                    'params': order_params,
                    'result': order_result,
                    'entry_time': datetime.now(),
                    'status': 'active'
                }
                
                logger.info(f"Order tracked: {order_id}")
                return order_id
            else:
                logger.warning("Order placement failed")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
            
    async def close_order(self, order_executor, order_id):
        """
        Close an active order
        
        Args:
            order_executor: OrderExecutor instance
            order_id: ID of the order to close
        """
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found")
                return False
                
            # Execute order closure
            closed = await order_executor.close_position(order_id)
            
            if closed:
                # Update order status
                self.active_orders[order_id]['status'] = 'closed'
                self.active_orders[order_id]['exit_time'] = datetime.now()
                
                # Move to history
                self.order_history.append(self.active_orders.pop(order_id))
                
                logger.info(f"Order closed: {order_id}")
                return True
            else:
                logger.warning(f"Failed to close order: {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing order: {str(e)}")
            return False
            
    def get_active_orders(self):
        """Return list of active orders"""
        return self.active_orders
        
    def get_order_history(self):
        """Return order history"""
        return self.order_history
        
    def calculate_performance_metrics(self):
        """Calculate performance metrics from order history"""
        try:
            total_trades = len(self.order_history)
            winning_trades = len([
                trade for trade in self.order_history
                if trade['result'].get('profit', 0) > 0
            ])
            
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'total_profit': sum(
                    trade['result'].get('profit', 0)
                    for trade in self.order_history
                )
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return None

