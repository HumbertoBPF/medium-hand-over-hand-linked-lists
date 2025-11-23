import threading
from datetime import datetime
from typing import Callable

import matplotlib.pyplot as plt

from linked_lists import LinkedList, SimpleLinkedList, StandardConcurrentLinkedList, HandOverHandLinkedList, \
    HybridHandOverhandLinkedList, MeasurementMixin

# Maximum number of threads to be used when comparing the performance of the linked lists
N_THREADS = 8
# Number of items to insert in the linked list before the lookup comparison
NUM_ITEMS = 10**6

def test_simple_linked_list():
    linked_list = SimpleLinkedList()

    linked_list.insert(key=1)
    linked_list.insert(key=2)
    linked_list.insert(key=3)

    print("Node 1 exists?", linked_list.lookup(key=1))
    print("Node 2 exists?", linked_list.lookup(key=2))
    print("Node 3 exists?", linked_list.lookup(key=3))
    print("Node 4 exists?", linked_list.lookup(key=4))

    print("Traverse the linked list:")
    linked_list.traversal()


def test_concurrent_inserts_linked_list(linked_list: LinkedList):
    # Should insert nodes from 0 to 9 to the linked list
    thread_1 = threading.Thread(target=insert_n_times, args=(linked_list, 10, 0))
    # Should insert nodes from 10 to 19 to the linked list
    thread_2 = threading.Thread(target=insert_n_times, args=(linked_list, 10, 10))

    thread_1.start()
    thread_2.start()

    thread_1.join()
    thread_2.join()

    print("Traverse the linked list:")
    linked_list.traversal()


def test_concurrent_inserts_at_linked_list(linked_list: LinkedList):
    # Should insert nodes from 10 to 19 into the linked list at the fifth position
    thread_1 = threading.Thread(target=insert_at_n_times, args=(linked_list, 10, 10, 5))
    # Should insert nodes from 20 to 29 into the linked list at the fifth position
    thread_2 = threading.Thread(target=insert_at_n_times, args=(linked_list, 10, 20, 5))

    thread_1.start()
    thread_2.start()

    thread_1.join()
    thread_2.join()

    print("Traverse the linked list:")
    linked_list.traversal()


def insert_at_n_times(linked_list: LinkedList, n: int, start_index: int, insertion_position: int):
    """
    Function that inserts into the linked list n times.
    :param linked_list: concerned linked list.
    :param n: number of insertions.
    :param start_index: starting index.
    :param insertion_position: insertion position.
    """
    for i in range(n):
        linked_list.insert_at(key=start_index + i, idx=insertion_position)


def lookup_n_times(linked_list: LinkedList, n: int):
    """
    Function that looks up in the linked list n times.
    :param linked_list: concerned linked list.
    :param n: number of lookups.
    """
    for i in range(n):
        linked_list.lookup(key=-1)


def insert_n_times(linked_list: LinkedList, n: int, start_index: int):
    """
    Function that inserts into the linked list n times.
    :param linked_list: concerned linked list.
    :param n: number of insertions.
    :param start_index: starting index.
    """
    for i in range(n):
        linked_list.insert(key=start_index + i)


# Removes the time measurements from this function since they will be performed by the MeasurementMixin
def start_threads(threads: list[threading.Thread]):
    """
    Triggers all the threads provided in the input list.
    :param threads: an array of threading.Thread objects.
    """
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def initialize_linked_list(linked_list: LinkedList, n: int) -> LinkedList:
    """
    Initialize linked list by inserting values from 0 to n - 1 into it.
    :param linked_list: linked list to be initialized.
    :param n: number of items to be inserted.
    :return: the initialized linked list.
    """
    for i in range(n):
        linked_list.insert(key=i)
    return linked_list


def insert_in_threads(linked_list: LinkedList, n_threads: int) -> float:
    """
    This function inserts into the provided linked list in n threads (100k times each).
    :param linked_list: linked list to be used
    :param n_threads: number of threads to be spawned
    :return: the time it takes to complete the insert operation in all involved threads.
    """
    # Empty linked list
    linked_list.clear()
    threads = []

    # Reset measurement mixin
    linked_list.reset_measurement()

    for i in range(n_threads):
        threads.append(threading.Thread(target=insert_n_times, args=(linked_list, 1, 2 * i)))

    # Call MeasurementMixin.measure_time_to_run to measure how long all threads take to execute
    return MeasurementMixin.measure_time_to_run(lambda : start_threads(threads))


def lookup_in_threads(linked_list: LinkedList, n_threads: int) -> float:
    """
    This function lookups into the provided linked list in n threads (100k times each).
    :param linked_list: linked list to be used
    :param n_threads: number of threads to be spawned
    :return: the time it takes to complete the insert operation in all involved threads.
    """
    threads = []

    # Reset measurement mixin
    linked_list.reset_measurement()

    for i in range(n_threads):
        threads.append(threading.Thread(target=lookup_n_times, args=(linked_list, 1)))

    # Call MeasurementMixin.measure_time_to_run to measure how long all threads take to execute
    return MeasurementMixin.measure_time_to_run(lambda : start_threads(threads))


def get_lookup_response_time(linked_list: LinkedList, n_threads: int) -> float:
    """
    This function lookups into the provided linked list in n threads (100k times each). It returns the "response time":
    the time it takes for the threads to start looking up the linked list (i.e., to check the first item of the linked
    list).
    :param linked_list: linked list to be used
    :param n_threads: number of threads to be spawned
    :return: the "response time".
    """
    threads = []

    # Reset measurement mixin
    linked_list.reset_measurement()

    for i in range(n_threads):
        threads.append(threading.Thread(target=lookup_n_times, args=(linked_list, 1)))

    start_threads(threads)
    response_time = linked_list.response_time.total_seconds()
    print("Response time =", response_time)
    return response_time


def build_and_show_comparative_plot(
    n_threads_array: list[float],
    t_linked_list_1_array: list[float],
    linked_list_1_label: str,
    t_linked_list_2_array: list[float],
    linked_list_2_label: str
):
    """
    Builds a comparative plot.
    :param n_threads_array: array to be used to the x-axis
    :param t_linked_list_1_array: array to be used for the plot in blue.
    :param linked_list_1_label: label to be used for the plot in blue.
    :param t_linked_list_2_array: array to be used for the plot in red.
    :param linked_list_2_label: label to be used for the plot in red.
    """
    fig, ax = plt.subplots()
    ax.plot(n_threads_array, t_linked_list_1_array, color="blue", label=linked_list_1_label)
    ax.plot(n_threads_array, t_linked_list_2_array, color="red", label=linked_list_2_label)
    ax.legend()
    ax.set_title(f"{linked_list_1_label} vs {linked_list_2_label}")
    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Time (seconds)")
    plt.show()


def compare_linked_lists_and_plot(
    multithread_func: Callable[[LinkedList, int], float],
    linked_list_1: LinkedList,
    linked_list_2: LinkedList,
    linked_list_1_label: str,
    linked_list_2_label: str,
):
    """
    This function runs the provided function for both linked lists. A plot with the outcome of the comparison
    is output at the end.
    :param multithread_func: a multi-thread function to run for both linked lists. It must receive a linked
    list as an input and the number of threads to be used.
    :param linked_list_1: first linked list
    :param linked_list_2: second linked list
    :param linked_list_1_label: label for the first linked list in the generated plot
    :param linked_list_2_label: label for the second linked list in the generated plot
    """
    n_threads_array = []
    t_linked_list_1_array = []
    t_linked_list_2_array = []

    for n_threads in range(1, N_THREADS + 1):
        print("=============================================================================")
        n_threads_array.append(n_threads)

        t_linked_list_1 = multithread_func(linked_list_1, n_threads)
        t_linked_list_2 = multithread_func(linked_list_2, n_threads)

        t_linked_list_1_array.append(t_linked_list_1)
        t_linked_list_2_array.append(t_linked_list_2)

    build_and_show_comparative_plot(
        n_threads_array=n_threads_array,
        t_linked_list_1_array=t_linked_list_1_array,
        linked_list_1_label=linked_list_1_label,
        t_linked_list_2_array=t_linked_list_2_array,
        linked_list_2_label=linked_list_2_label,
    )


def compare_simple_and_standard_concurrent_safe_linked_lists():
    linked_list_1_label = "Simple Linked List"
    linked_list_2_label = "Concurrent-Safe Linked List"

    print("Compare inserts")

    compare_linked_lists_and_plot(
        multithread_func=insert_in_threads,
        linked_list_1=initialize_linked_list(SimpleLinkedList(), NUM_ITEMS),
        linked_list_2=initialize_linked_list(StandardConcurrentLinkedList(), NUM_ITEMS),
        linked_list_1_label=linked_list_1_label,
        linked_list_2_label=linked_list_2_label,
    )

    print("Compare lookups")

    compare_linked_lists_and_plot(
        multithread_func=lookup_in_threads,
        linked_list_1=initialize_linked_list(SimpleLinkedList(), NUM_ITEMS),
        linked_list_2=initialize_linked_list(StandardConcurrentLinkedList(), NUM_ITEMS),
        linked_list_1_label=linked_list_1_label,
        linked_list_2_label=linked_list_2_label,
    )

    print("Compare response time")

    compare_linked_lists_and_plot(
        multithread_func=get_lookup_response_time,
        linked_list_1=initialize_linked_list(SimpleLinkedList(), NUM_ITEMS),
        linked_list_2=initialize_linked_list(StandardConcurrentLinkedList(), NUM_ITEMS),
        linked_list_1_label=linked_list_1_label,
        linked_list_2_label=linked_list_2_label,
    )


def compare_standard_concurrent_safe_and_hand_over_hand_linked_lists():
    linked_list_1_label = "Standard Concurrent-Safe Linked List"
    linked_list_2_label = "Hand-Over-Hand Linked List"

    print("Compare inserts")

    compare_linked_lists_and_plot(
        multithread_func=insert_in_threads,
        linked_list_1=initialize_linked_list(StandardConcurrentLinkedList(), NUM_ITEMS),
        linked_list_2=initialize_linked_list(HandOverHandLinkedList(), NUM_ITEMS),
        linked_list_1_label=linked_list_1_label,
        linked_list_2_label=linked_list_2_label,
    )

    print("Compare lookups")

    compare_linked_lists_and_plot(
        multithread_func=lookup_in_threads,
        linked_list_1=initialize_linked_list(StandardConcurrentLinkedList(), NUM_ITEMS),
        linked_list_2=initialize_linked_list(HandOverHandLinkedList(), NUM_ITEMS),
        linked_list_1_label=linked_list_1_label,
        linked_list_2_label=linked_list_2_label,
    )

    print("Compare response time")

    compare_linked_lists_and_plot(
        multithread_func=get_lookup_response_time,
        linked_list_1=initialize_linked_list(StandardConcurrentLinkedList(), NUM_ITEMS),
        linked_list_2=initialize_linked_list(HandOverHandLinkedList(), NUM_ITEMS),
        linked_list_1_label=linked_list_1_label,
        linked_list_2_label=linked_list_2_label,
    )


def compare_hand_over_hand_and_hybrid_hand_over_hand_linked_lists():
    linked_list_1_label = "Hand-Over-Hand Linked List"
    linked_list_2_label = "Hybrid Hand-Over-Hand Linked List"

    print("Compare inserts")

    compare_linked_lists_and_plot(
        multithread_func=insert_in_threads,
        linked_list_1=initialize_linked_list(HandOverHandLinkedList(), NUM_ITEMS),
        linked_list_2=initialize_linked_list(HybridHandOverhandLinkedList(capacity=100), NUM_ITEMS),
        linked_list_1_label=linked_list_1_label,
        linked_list_2_label=linked_list_2_label,
    )

    print("Compare lookups")

    compare_linked_lists_and_plot(
        multithread_func=lookup_in_threads,
        linked_list_1=initialize_linked_list(HandOverHandLinkedList(), NUM_ITEMS),
        linked_list_2=initialize_linked_list(HybridHandOverhandLinkedList(capacity=100), NUM_ITEMS),
        linked_list_1_label=linked_list_1_label,
        linked_list_2_label=linked_list_2_label,
    )

    print("Compare response time")

    compare_linked_lists_and_plot(
        multithread_func=get_lookup_response_time,
        linked_list_1=initialize_linked_list(HandOverHandLinkedList(), NUM_ITEMS),
        linked_list_2=initialize_linked_list(HybridHandOverhandLinkedList(capacity=100), NUM_ITEMS),
        linked_list_1_label=linked_list_1_label,
        linked_list_2_label=linked_list_2_label,
    )


def hybrid_hand_over_hand_linked_list_performance_vs_capacity():
    capacity_array = []
    lookup_time_array = []
    response_time_array = []

    for n in range(0, 21):
        capacity = 2 ** n

        linked_list_for_lookup = initialize_linked_list(linked_list=HybridHandOverhandLinkedList(capacity=capacity), n=NUM_ITEMS)
        lookup_time = lookup_in_threads(linked_list=linked_list_for_lookup, n_threads=8)

        linked_list_for_response_time = initialize_linked_list(linked_list=HybridHandOverhandLinkedList(capacity=capacity), n=NUM_ITEMS)
        response_time = get_lookup_response_time(linked_list=linked_list_for_response_time, n_threads=8)

        capacity_array.append(capacity)
        lookup_time_array.append(lookup_time)
        response_time_array.append(response_time)

    fig, ax = plt.subplots()
    ax.plot(capacity_array, lookup_time_array, color="red", label="Lookup time varying capacity")
    ax.legend()
    ax.set_title("Lookup time varying capacity of a hybrid hand-over-hand linked list")
    ax.set_xlabel("Capacity")
    ax.set_ylabel("Lookup time (seconds)")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(capacity_array, response_time_array, color="red", label="Response time varying capacity")
    ax.legend()
    ax.set_title("Response time varying capacity of a hybrid hand-over-hand linked list")
    ax.set_xlabel("Capacity")
    ax.set_ylabel("Response time (seconds)")
    plt.show()


def find_item(target: int, linked_list: LinkedList):
    return linked_list.lookup(key=target)


if __name__ == "__main__":
    compare_simple_and_standard_concurrent_safe_linked_lists()
    compare_standard_concurrent_safe_and_hand_over_hand_linked_lists()
    compare_hand_over_hand_and_hybrid_hand_over_hand_linked_lists()

    hybrid_hand_over_hand_linked_list_performance_vs_capacity()

    linked_list = initialize_linked_list(linked_list=SimpleLinkedList(), n=10)
    test_concurrent_inserts_at_linked_list(linked_list=linked_list)
    print("\n====================================================================")

    linked_list = initialize_linked_list(linked_list=StandardConcurrentLinkedList(), n=10)
    test_concurrent_inserts_at_linked_list(linked_list=linked_list)
    print("\n====================================================================")

    linked_list = initialize_linked_list(linked_list=HandOverHandLinkedList(), n=10)
    test_concurrent_inserts_at_linked_list(linked_list=linked_list)
    print("\n====================================================================")

    linked_list = initialize_linked_list(linked_list=HybridHandOverhandLinkedList(capacity=5), n=10)
    test_concurrent_inserts_at_linked_list(linked_list=linked_list)
    print("\n====================================================================")

    linked_list = initialize_linked_list(linked_list=HybridHandOverhandLinkedList(capacity=3), n=10)
    test_concurrent_inserts_at_linked_list(linked_list=linked_list)
    print("\n====================================================================")
