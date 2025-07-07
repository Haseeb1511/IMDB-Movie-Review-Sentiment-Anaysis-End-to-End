import logging
import sys

def error_message_detail(error: Exception, error_detail: sys):
    """
    Formats detailed error information with filename and line number,
    and logs it using Python's logging module.
    """

    # Get traceback details of the current exception
    _, _, exc_tb = error_detail.exc_info()

    # Get the filename where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Get the line number where the exception occurred
    line_number = exc_tb.tb_lineno

    # Build the full error message
    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number [{line_number}]: {str(error)}"
    )

    # Log the error message
    logging.error(error_message)

    return error_message

class MyException(Exception):

    def __init__(self, error_message:str,error_detail:sys):
        
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


