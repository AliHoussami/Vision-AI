import logging

# The library modules log through logging.getLogger("footfall.*"). Nothing
# in the suite asserts on log output, so keep it quiet -- a failing test's
# captured stderr stays about the failure, not backoff/reconnect chatter.
logging.getLogger("footfall").setLevel(logging.CRITICAL)
