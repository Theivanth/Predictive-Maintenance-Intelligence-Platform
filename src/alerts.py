def generate_alert(status):

    if status == "Healthy":

        return {
            "level": "NORMAL",
            "message": "Machine operating normally.",
        }

    if status == "Warning":

        return {
            "level": "WARNING",
            "message": "Inspection recommended.",
        }

    if status == "Critical":

        return {
            "level": "CRITICAL",
            "message": "Immediate maintenance required.",
        }

    return {
        "level": "UNKNOWN",
        "message": "Unknown machine status.",
    }