"""A tiny in-memory store with the same put/get/search calls as the LangGraph store the teacher service uses."""


class Item:
    def __init__(self, key, value):
        self.key, self.value = key, value


class FakeStore:
    def __init__(self):
        self.data: dict = {}

    def put(self, ns, key, value):
        self.data[(tuple(ns), key)] = value

    def get(self, ns, key):
        return Item(key, self.data[(tuple(ns), key)]) if (tuple(ns), key) in self.data else None

    def search(self, ns, limit=10, offset=0, **_):
        items = [Item(k, v) for (n, k), v in self.data.items() if n == tuple(ns)]
        return items[offset: offset + limit]
