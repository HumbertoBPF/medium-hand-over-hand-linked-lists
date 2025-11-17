import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class MeasurementMixin:
    """
    Class with helper methods to obtain measurements regarding the execution time of the linked lists.
    """
    def __init__(self):
        # We keep a set to save the threads that have been already been tracked
        self.threads = set()
        # We need to define an initial time to have it as a reference
        self.initial_time = datetime.now()
        self.response_time = self.initial_time - self.initial_time

    def measure(self):
        """
        Method to help measure the response time. The response time is defined as the time it takes for
        all the threads start running.
        """
        thread_id = threading.get_ident()

        # Only updates the response time if the thread has not been taken into account yet
        if thread_id not in self.threads:
            self.response_time = max(datetime.now() - self.initial_time, self.response_time)
            self.threads.add(thread_id)

    @staticmethod
    def measure_time_to_run(func):
        """
        Measures the execution time of the provided function.
        :param func: function to measure the execution time.
        :return: the time the function takes to complete.
        """
        initial_time = datetime.now()

        func()

        final_time = datetime.now()
        delta_t_in_seconds = (final_time - initial_time).total_seconds()
        print("Total time:", delta_t_in_seconds)

        return delta_t_in_seconds

    def reset_measurement(self):
        """
        Clears the measures.
        """
        self.threads.clear()
        self.initial_time = datetime.now()
        self.response_time = self.initial_time - self.initial_time


class LinkedListNode:
    def __init__(self, key: int):
        self.head: int = key
        self.next: Optional[LinkedListNode] = None


class LinkedList(ABC, MeasurementMixin):
    def __init__(self):
        super().__init__()
        self.head: Optional[LinkedListNode] = None
        self.length: int = 0

    @abstractmethod
    def lookup(self, key: int) -> bool:
        """
        Searches for a node with the given key.
        :param key: key to be searched on the linked list
        :return: a boolean indicating whether the node is present in the list
        """
        pass

    @abstractmethod
    def insert(self, key: int):
        """
        Inserts a node with the provided key at the beginning (head) of the linked list.
        :param key: key to be inserted into the list.
        """
        pass

    @abstractmethod
    def insert_at(self, key: int, idx: int) -> bool:
        """
        Inserts a node with the provided key at the specified position of the linked list.
        :param key: key to be inserted into the list.
        :param idx: the insertion position.
        :return: a boolean indicating whether the operation succeeded.
        """
        pass

    @abstractmethod
    def traversal(self):
        """
        Traverses the linked list printing all its nodes.
        """
        pass

    def clear(self):
        self.head = None
        self.length = 0


class SimpleLinkedList(LinkedList):
    def lookup(self, key: int) -> bool:
        current_node = self.head

        while current_node is not None:
            # Measure time the first iteration effectively starts
            self.measure()

            if current_node.head == key:
                return True
            current_node = current_node.next

        return False

    def insert(self, key: int):
        new_node = LinkedListNode(key=key)
        old_head = self.head
        new_node.next = old_head
        # Delay to force race condition
        # time.sleep(0.1)
        self.head = new_node
        self.length += 1

    def insert_at(self, key: int, idx: int) -> bool:
        current_node = self.head
        current_idx = 0

        while current_node is not None:
            # Measure time the first iteration effectively starts
            self.measure()

            if current_idx == idx:
                # Insert the new node between the previous and the current node
                new_node = LinkedListNode(key=key)
                new_node.next = current_node.next
                # Delay to force race condition
                time.sleep(0.1)
                current_node.next = new_node
                self.length += 1
                return True

            current_node = current_node.next
            current_idx += 1

        return False

    def traversal(self):
        current_node = self.head

        while current_node is not None:
            print(current_node.head, end=" -> ")
            current_node = current_node.next


class StandardConcurrentLinkedList(SimpleLinkedList):
    def __init__(self):
        super().__init__()
        self.lock: threading.Lock = threading.Lock()

    def lookup(self, key: int, disable_locking = False) -> bool:
        if not disable_locking:
            self.lock.acquire()
        found = super().lookup(key=key)
        if not disable_locking:
            self.lock.release()
        return found

    def insert(self, key: int, disable_locking = False):
        if not disable_locking:
            self.lock.acquire()
        super().insert(key=key)
        if not disable_locking:
            self.lock.release()

    def insert_at(self, key: int, idx: int, disable_locking = False) -> bool:
        if not disable_locking:
            self.lock.acquire()
        inserted = super().insert_at(key=key, idx=idx)
        if not disable_locking:
            self.lock.release()
        return inserted

    def traversal(self, disable_locking = False):
        if not disable_locking:
            self.lock.acquire()
        super().traversal()
        if not disable_locking:
            self.lock.release()


class ConcurrentLinkedListNode(LinkedListNode):
    def __init__(self, key: int):
        super().__init__(key=key)
        self.next: Optional[ConcurrentLinkedListNode] = None
        self.lock: threading.Lock = threading.Lock()


class HandOverHandLinkedList(SimpleLinkedList):
    def __init__(self) -> None:
        super().__init__()
        self.head: Optional[ConcurrentLinkedListNode] = None
        self.head_lock = threading.Lock()  # Lock for operations involving the head

    def lookup(self, key: int) -> bool:
        current_node = self.head
        prev_node = None

        while current_node is not None:
            # Acquire the lock of the current node
            current_node.lock.acquire()

            # Measure time the first iteration effectively starts
            self.measure()

            # Release the lock of the previous node
            if prev_node is not None:
                prev_node.lock.release()

            if current_node.head == key:
                # Release the lock of the current node before returning to avoid deadlocking
                current_node.lock.release()
                return True

            prev_node = current_node
            current_node = current_node.next

        # Release the lock of the previous node
        if prev_node is not None:
            prev_node.lock.release()
        return False

    def insert(self, key: int):
        new_node = ConcurrentLinkedListNode(key=key)
        # Lock the head
        self.head_lock.acquire()
        old_head = self.head
        # Add the new node to the linked list
        new_node.next = old_head
        self.head = new_node
        self.length += 1
        # Release the head
        self.head_lock.release()

    def insert_at(self, key: int, idx: int, disable_locking = False) -> bool:
        current_node = self.head
        prev_node = None
        current_idx = 0

        while current_node is not None:
            # Acquire the lock of the current node
            current_node.lock.acquire()

            # Measure time the first iteration effectively starts
            self.measure()

            # Release the lock of the previous node
            if prev_node is not None:
                prev_node.lock.release()

            if current_idx == idx:
                # Insert the new node between the previous and the current node
                new_node = ConcurrentLinkedListNode(key=key)
                new_node.next = current_node.next
                # Delay to force race condition
                time.sleep(0.1)
                current_node.next = new_node
                self.length += 1
                current_node.lock.release()
                return True

            prev_node = current_node
            current_node = current_node.next
            current_idx += 1

        # Release the lock of the previous node
        if prev_node is not None:
            prev_node.lock.release()
        return False

    def traversal(self):
        super().traversal()


class SubLinkedList(LinkedListNode):
    def __init__(self, key: StandardConcurrentLinkedList):
        # For the hybrid approach, the entire linked list will be split in smaller standard concurrent-safe linked lists.
        self.head: StandardConcurrentLinkedList = key
        self.next: Optional[StandardConcurrentLinkedList] = None


class HybridHandOverhandLinkedList(LinkedList):
    def __init__(self, capacity: int):
        super().__init__()
        # In this hybrid approach, the list holds a lock at every "capacity" number of items.
        # This is the same as having a dedicated standard linked list after every "capacity" number of items.
        self.head: Optional[SubLinkedList] = None
        self.length: int = 0
        self.capacity: int = capacity       # How many items each sub-linked list must contain in full capacity
        self.head_lock = threading.Lock()   # Lock for operations involving the head

    def lookup(self, key: int) -> bool:
        current_node = self.head
        prev_node: Optional[SubLinkedList] = None
        found = False

        while current_node is not None:
            # Lock the sub-linked list
            current_node.head.lock.acquire()

            # Measure time the first iteration effectively starts
            self.measure()

            # Release the previous sub-linked list
            if prev_node is not None:
                prev_node.head.lock.release()

            # Search the key in the sub-linked list
            found = current_node.head.lookup(key=key, disable_locking=True)

            # If the item is found, we do not need to proceed any further
            if found:
                current_node.head.lock.release()
                return found

            # Move to the next sub-linked list if the item has not been found yet
            prev_node = current_node
            current_node = current_node.next

        # Release the last sub-linked list
        if prev_node is not None:
            prev_node.head.lock.release()
        return found

    def insert(self, key: int):
        self.head_lock.acquire()

        # Linked list is empty, so we must initialize the linked list by allocating
        # a standard concurrent linked list and placing it at the head
        if self.head is None:
            sub_linked_list = StandardConcurrentLinkedList()
            sub_linked_list.insert(key=key, disable_locking=True)
            self.head = SubLinkedList(key=sub_linked_list)
            self.length += 1
            self.head_lock.release()
            return

        # The first sub-linked list has reached maximum capacity,
        # so we need to allocate a new standard concurrent linked list
        # and make it point to the fist sub-linked list (previous head)
        if self.head.head.length == self.capacity:
            sub_linked_list = StandardConcurrentLinkedList()
            sub_linked_list.insert(key=key, disable_locking=True)
            old_head = self.head
            self.head = SubLinkedList(key=sub_linked_list)
            self.head.next = old_head
            self.length += 1
            self.head_lock.release()
            return

        # If the first sub-linked list has not reached the maximum capacity, just insert the key into it
        self.head.head.insert(key=key, disable_locking=True)
        self.length += 1
        self.head_lock.release()

    def insert_at(self, key: int, idx: int) -> bool:
        current_node = self.head
        prev_node: Optional[SubLinkedList] = None
        current_idx = 0
        inserted = False

        while current_node is not None:
            # Lock the sub-linked list
            current_node.head.lock.acquire()

            # Measure time the first iteration effectively starts
            self.measure()

            # Release the previous sub-linked list
            if prev_node is not None:
                prev_node.head.lock.release()

            # Search the key in the sub-linked list
            inserted = current_node.head.insert_at(key=key, idx=idx-current_idx, disable_locking=True)

            # If the item is found, we do not need to proceed any further
            if inserted:
                current_node.head.lock.release()
                return inserted

            # Move to the next sub-linked list if the item has not been found yet
            prev_node = current_node
            current_idx += current_node.head.length
            current_node = current_node.next

        # Release the last sub-linked list
        if prev_node is not None:
            prev_node.head.lock.release()
        return inserted

    def traversal(self):
        current_node = self.head
        prev_node: Optional[SubLinkedList] = None

        while current_node is not None:
            # Lock the sub-linked list
            current_node.head.lock.acquire()
            # Release the previous sub-linked list
            if prev_node is not None:
                prev_node.head.lock.release()
            current_node.head.traversal(disable_locking=True)
            # Move to the next sub-linked list
            prev_node = current_node
            current_node = current_node.next

        # Release the last sub-linked list
        if prev_node is not None:
            prev_node.head.lock.release()
        return False