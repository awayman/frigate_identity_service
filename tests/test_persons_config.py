"""Unit tests for persons configuration loading and validation."""
import pytest
import tempfile
import os
import sys
import time
import yaml
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions to test - we need to do this carefully to avoid running the main code
import importlib.util

def get_module_functions():
    """Load the identity_service module and extract needed functions."""
    spec = importlib.util.spec_from_file_location(
        "identity_service_test",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "identity_service.py")
    )
    # We can't directly import because it has initialization code that runs
    # Instead we'll test the functions in isolation
    return spec

@pytest.fixture
def temp_persons_file():
    """Create a temporary persons.yaml file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)

@pytest.fixture
def valid_config():
    """Return a valid persons configuration."""
    return {
        "alice": {
            "name": "Alice",
            "display_name": "Alice Johnson",
            "role": "child",
            "age": 7,
            "supervision_required": True
        },
        "bob": {
            "name": "Bob",
            "display_name": "Bob Johnson",
            "role": "adult",
            "age": 35,
            "supervision_required": False
        }
    }

@pytest.fixture
def invalid_config_no_name():
    """Return an invalid config with missing name."""
    return {
        "charlie": {
            "role": "child",
            "age": 10
        }
    }

@pytest.fixture
def invalid_config_bad_role():
    """Return a config with invalid role."""
    return {
        "david": {
            "name": "David",
            "role": "teenager",  # Invalid role
            "age": 15
        }
    }

@pytest.fixture
def invalid_config_bad_age():
    """Return a config with invalid age."""
    return {
        "eve": {
            "name": "Eve",
            "age": -5  # Invalid age
        }
    }

class TestValidatePersonsConfig:
    """Test validate_persons_config function."""
    
    def test_valid_config(self, valid_config, capsys):
        """Test validation of a valid configuration."""
        from identity_service import validate_persons_config
        
        result = validate_persons_config(valid_config)
        assert result is True
        
        captured = capsys.readouterr()
        assert "ERROR" not in captured.out
    
    def test_empty_config(self, capsys):
        """Test validation of an empty configuration."""
        from identity_service import validate_persons_config
        
        result = validate_persons_config({})
        assert result is True
        
        captured = capsys.readouterr()
        assert "WARNING: Persons config is empty" in captured.out
    
    def test_invalid_type(self, capsys):
        """Test validation with non-dict input."""
        from identity_service import validate_persons_config
        
        result = validate_persons_config("not a dict")
        assert result is False
        
        captured = capsys.readouterr()
        assert "ERROR: Persons config must be a dictionary" in captured.out
    
    def test_config_without_name(self, invalid_config_no_name, capsys):
        """Test validation catches missing name field."""
        from identity_service import validate_persons_config
        
        result = validate_persons_config(invalid_config_no_name)
        assert result is True  # Still valid, just warnings
        
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "missing required 'name' or 'display_name'" in captured.out
    
    def test_config_with_invalid_role(self, invalid_config_bad_role, capsys):
        """Test validation catches invalid role."""
        from identity_service import validate_persons_config
        
        result = validate_persons_config(invalid_config_bad_role)
        assert result is True  # Still valid, just warnings
        
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "invalid role 'teenager'" in captured.out
    
    def test_config_with_invalid_age(self, invalid_config_bad_age, capsys):
        """Test validation catches invalid age."""
        from identity_service import validate_persons_config
        
        result = validate_persons_config(invalid_config_bad_age)
        assert result is True  # Still valid, just warnings
        
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "invalid age" in captured.out
    
    def test_valid_roles(self, capsys):
        """Test that all valid roles pass validation."""
        from identity_service import validate_persons_config
        
        for role in ["child", "adult", "unknown"]:
            config = {"person": {"name": "Test", "role": role}}
            result = validate_persons_config(config)
            assert result is True
        
        captured = capsys.readouterr()
        assert "invalid role" not in captured.out

class TestLoadPersonsConfig:
    """Test load_persons_config function."""
    
    def test_load_valid_config(self, temp_persons_file, valid_config, capsys):
        """Test loading a valid configuration file."""
        from identity_service import load_persons_config
        
        # Write valid config to file
        with open(temp_persons_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        result = load_persons_config(temp_persons_file)
        
        assert result == valid_config
        assert len(result) == 2
        assert "alice" in result
        assert "bob" in result
        
        captured = capsys.readouterr()
        assert "Successfully loaded persons config with 2 entries" in captured.out
    
    def test_load_missing_file(self, capsys):
        """Test loading a non-existent file."""
        from identity_service import load_persons_config
        
        result = load_persons_config("/nonexistent/path/persons.yaml")
        
        assert result == {}
        captured = capsys.readouterr()
        assert "WARNING: Persons config file not found" in captured.out
    
    def test_load_empty_file(self, temp_persons_file, capsys):
        """Test loading an empty file."""
        from identity_service import load_persons_config
        
        # Create empty file
        with open(temp_persons_file, 'w') as f:
            f.write("")
        
        result = load_persons_config(temp_persons_file)
        
        assert result == {}
        captured = capsys.readouterr()
        assert "WARNING: Persons config file" in captured.out
        assert "is empty" in captured.out
    
    def test_load_malformed_yaml(self, temp_persons_file, capsys):
        """Test loading a malformed YAML file."""
        from identity_service import load_persons_config
        
        # Write malformed YAML
        with open(temp_persons_file, 'w') as f:
            f.write("invalid: yaml: content: [\n  unclosed bracket")
        
        result = load_persons_config(temp_persons_file)
        
        assert result == {}
        captured = capsys.readouterr()
        assert "ERROR: Failed to parse YAML" in captured.out
    
    def test_load_config_with_comments(self, temp_persons_file, capsys):
        """Test loading a config file with comments."""
        from identity_service import load_persons_config
        
        # Write config with comments
        with open(temp_persons_file, 'w') as f:
            f.write("""
# This is a comment
alice:
  name: Alice  # Inline comment
  role: child
  age: 7
""")
        
        result = load_persons_config(temp_persons_file)
        
        assert len(result) == 1
        assert "alice" in result
        assert result["alice"]["name"] == "Alice"
        assert result["alice"]["age"] == 7

class TestHotReload:
    """Test hot-reload functionality."""
    
    def test_reload_persons_config(self, temp_persons_file, valid_config, capsys):
        """Test reloading configuration."""
        from identity_service import load_persons_config, reload_persons_config
        import identity_service
        
        # Write initial config
        with open(temp_persons_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        # Set up module state
        identity_service.PERSONS_CONFIG_PATH = temp_persons_file
        identity_service.persons_config = {}
        
        # Reload
        reload_persons_config()
        
        assert len(identity_service.persons_config) == 2
        captured = capsys.readouterr()
        assert "INFO: Persons config reloaded successfully" in captured.out
    
    @patch('identity_service.reload_persons_config')
    def test_file_handler_on_modify(self, mock_reload, temp_persons_file):
        """Test file system event handler triggers reload."""
        from identity_service import PersonsConfigFileHandler
        from watchdog.events import FileModifiedEvent
        
        handler = PersonsConfigFileHandler(temp_persons_file)
        
        # Create a file modification event
        event = FileModifiedEvent(temp_persons_file)
        handler.on_modified(event)
        
        # Should have called reload
        mock_reload.assert_called_once()
    
    def test_file_handler_ignores_directories(self, temp_persons_file):
        """Test file system event handler ignores directory events."""
        from identity_service import PersonsConfigFileHandler
        from watchdog.events import DirModifiedEvent
        
        handler = PersonsConfigFileHandler(temp_persons_file)
        
        # Create a directory modification event
        event = DirModifiedEvent(temp_persons_file)
        
        # Should not raise any errors
        handler.on_modified(event)
    
    def test_file_handler_ignores_other_files(self, temp_persons_file):
        """Test file system event handler ignores other files."""
        from identity_service import PersonsConfigFileHandler
        from watchdog.events import FileModifiedEvent
        import identity_service
        
        handler = PersonsConfigFileHandler(temp_persons_file)
        
        # Save initial config
        initial_config = identity_service.persons_config.copy()
        
        # Create an event for a different file
        event = FileModifiedEvent("/some/other/file.yaml")
        handler.on_modified(event)
        
        # Config should not have changed
        assert identity_service.persons_config == initial_config

class TestPublishIdentityEvent:
    """Test that person config is included in identity events."""
    
    def test_publish_with_person_config(self, valid_config):
        """Test publishing identity event with person config."""
        from identity_service import publish_identity_event
        import identity_service
        import json
        
        # Set up module state
        identity_service.persons_config = valid_config
        identity_service.FRIGATE_HOST = "http://localhost:5000"
        
        # Create mock MQTT client
        mock_client = Mock()
        
        # Call publish_identity_event
        publish_identity_event(
            mock_client,
            person_id="alice",
            camera="backyard",
            confidence=0.95,
            source="facial_recognition",
            zones=["pool"],
            event_id="test-event-123",
            timestamp=1234567890.0
        )
        
        # Verify publish was called
        assert mock_client.publish.called
        call_args = mock_client.publish.call_args
        
        # Extract the published message
        topic = call_args[0][0]
        payload = call_args[0][1]
        
        assert topic == "identity/person/alice"
        
        # Parse the JSON payload
        data = json.loads(payload)
        
        # Check person_config is included
        assert "person_config" in data
        assert data["person_config"]["display_name"] == "Alice Johnson"
        assert data["person_config"]["role"] == "child"
        assert data["person_config"]["age"] == 7
        assert data["person_config"]["supervision_required"] is True
    
    def test_publish_without_person_config(self):
        """Test publishing identity event for unknown person."""
        from identity_service import publish_identity_event
        import identity_service
        import json
        
        # Set up module state with empty config
        identity_service.persons_config = {}
        identity_service.FRIGATE_HOST = "http://localhost:5000"
        
        # Create mock MQTT client
        mock_client = Mock()
        
        # Call publish_identity_event for unknown person
        publish_identity_event(
            mock_client,
            person_id="unknown_person",
            camera="backyard",
            confidence=0.75,
            source="reid_model",
            zones=["yard"],
            event_id="test-event-456",
            timestamp=1234567890.0
        )
        
        # Verify publish was called
        assert mock_client.publish.called
        call_args = mock_client.publish.call_args
        
        # Extract the published message
        payload = call_args[0][1]
        data = json.loads(payload)
        
        # Check person_config is NOT included
        assert "person_config" not in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
