from enum import Enum
class Status(Enum):
    """
        Search Engine Model status enumeration:
        - DOWN: Not initilized.
        - PREPARING: Getting everything ready.
        - READY: Dataset ready to be used. Search engine ready for queries.
        - RANKING: Currently processing a search.
    """
    DOWN = 'down'
    PREPARING = 'preparing'
    READY = 'ready'
    RANKING = 'ranking'

    def __str__(self):
        return str(self.value)
