import os
import yaml
import torch
import logging

class ConfigManager:
    """Configuration manager for the person re-identification system"""
    
    def __init__(self, config_path):
        """Initialize the configuration manager
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.runtime_config = self.config.copy()
        self._configure_device()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _configure_device(self):
        """Configure the device (CPU/GPU) based on availability"""
        if self.config['system']['device'] == 'auto':
            if torch.cuda.is_available():
                self.runtime_config['system']['device'] = 'cuda:0'
                # Adjust batch size based on GPU memory
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if gpu_mem < 4:
                    self.runtime_config['system']['batch_size'] = 2
                elif gpu_mem > 8:
                    self.runtime_config['system']['batch_size'] = 8
            else:
                self.runtime_config['system']['device'] = 'cpu'
                # Adjust for CPU processing
                self.runtime_config['detection']['half_precision'] = False
                self.runtime_config['system']['batch_size'] = 2
    
    def get_config(self):
        """Get the current runtime configuration
        
        Returns:
            dict: The current configuration
        """
        return self.runtime_config
    
    def update_config(self, updates):
        """Update configuration with new values
        
        Args:
            updates (dict): Dictionary with configuration updates
        """
        # Update nested dictionaries
        for section, values in updates.items():
            if section in self.runtime_config and isinstance(values, dict):
                self.runtime_config[section].update(values)
            else:
                self.runtime_config[section] = values 