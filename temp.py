from logger.meta import GET_METADATA, SET_METADATA



if __name__ == "__main__":
    meta = GET_METADATA()
    meta += ["max_delay"]
    SET_METADATA(meta)