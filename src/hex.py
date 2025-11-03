"""
Hex class representing a single hexagonal cell in the city grid.
"""

class Hex:
    """
    Represents a single hexagonal cell in the city grid.
    
    Attributes:
        hex_id (int): A unique identifier for the hexagonal cell.
        # Additional geojson parameters may be added here for rendering/spatial calculations
    """
    
    def __init__(self, hex_id: int):
        """
        Initialize a Hex object.
        
        Args:
            hex_id (int): A unique identifier for the hexagonal cell.
        """
        self.hex_id = hex_id
        #TODO: future implementation: should we add list of services that pass through this hex?
    
    def __repr__(self):
        return f"Hex(hex_id={self.hex_id})"
    
    def __eq__(self, other):
        if not isinstance(other, Hex):
            return False
        return self.hex_id == other.hex_id
    
    def __hash__(self):
        return hash(self.hex_id)
