"""Video discovery history: earlier searches are kept per teacher and reopened without searching again."""
from langgraph.store.memory import InMemoryStore

from rafeeq.teacher import video_history as vh

R = [{"video_id": "abc", "title": "Linked lists", "start_sec": 30}]


def test_save_list_get_and_same_question_replaces():
    store = InMemoryStore()
    first = vh.save(store, "t1", "Linked lists ", R, "r1")
    again = vh.save(store, "t1", "linked   LISTS", R + R, "r1")
    assert first == again                                   # same question, one entry
    searches = vh.list_searches(store, "t1")
    assert len(searches) == 1 and searches[0]["count"] == 2
    found = vh.get_search(store, "t1", first)
    assert found["query"] == "linked LISTS" and found["results"][0]["video_id"] == "abc"


def test_teachers_do_not_see_each_other_and_empty_results_are_not_kept():
    store = InMemoryStore()
    sid = vh.save(store, "t1", "stacks", R)
    assert vh.get_search(store, "t2", sid) is None and vh.list_searches(store, "t2") == []
    assert vh.save(store, "t1", "nothing found", []) is None


def test_only_the_newest_are_kept_and_delete_works():
    store = InMemoryStore()
    for i in range(vh.MAX_KEEP + 5):
        vh.save(store, "t1", f"question {i}", R)
    assert len(vh.list_searches(store, "t1")) == vh.MAX_KEEP
    sid = vh.list_searches(store, "t1")[0]["search_id"]
    assert vh.delete_search(store, "t1", sid) and not vh.delete_search(store, "t1", sid)
    assert vh.get_search(store, "t1", "not-an-id") is None


def test_room_filter_keeps_searches_without_a_room():
    store = InMemoryStore()
    vh.save(store, "t1", "in room one", R, "r1")
    vh.save(store, "t1", "in room two", R, "r2")
    vh.save(store, "t1", "no room", R, "")
    assert {s["query"] for s in vh.list_searches(store, "t1", "r1")} == {"in room one", "no room"}
