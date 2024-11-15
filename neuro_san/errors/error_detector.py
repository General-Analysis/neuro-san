from typing import List
from typing import Set

from neuro_san.errors.error_formatter_factory import ErrorFormatterFactory
from neuro_san.interfaces.error_formatter import ErrorFormatter


class ErrorDetector:
    """
    Detects errors in agent output.
    """

    def __init__(self, agent_name: str,
                 error_formatter_name: str = None,
                 system_error_fragments: List[str] = None,
                 agent_error_fragments: List[str] = None):
        """
        Constructor

        :param agent_name: The name of the agent
        :param error_formatter_name: An optional name for which type of ErrorFormatter to use.
        :param system_error_fragments: A list of strings that identify errors from the system.
                By default this is None.
        :param agent_error_fragments: A list of strings that identify errors from the agent.
                By default this is None.
        """
        self.agent_name: str = agent_name
        self.error_formatter_name: str = error_formatter_name

        self.error_fragments: Set[str] = set()
        if system_error_fragments is not None:
            self.error_fragments.update(system_error_fragments)
        if agent_error_fragments is not None:
            self.error_fragments.update(agent_error_fragments)

    def is_error(self, output: str) -> bool:
        """
        Determines whether the output from the agent should be considered an error.

        :param output: String output from an agent
        :return: True/False as to whether or not this output should be considered an error.
        """
        if output is None:
            return False

        for fragment in self.error_fragments:
            if fragment in output:
                return True

        return False

    def handle_error(self, output: str, details: str = None) -> str:
        """
        Handles potential error in agent output.

        :param output: String output from an agent
        :param details: An optional details string to send to an ErrorFormatter if necessary.
        :return: The same string if there was no error, or if there was an error
                a properly formatted error per the choice in the error_formatter_name.
        """
        if not self.is_error(output):
            return output

        formatter: ErrorFormatter = ErrorFormatterFactory.create_formatter(self.error_formatter_name)
        error_output = formatter.format(self.agent_name, output, details)

        return error_output
