type2color = {
    's': ' \033[95mSuccess:\033[0m ',
    'i': ' \033[94mInfo:\033[0m ',
    'd': ' \033[92mDebug:\033[0m ',
    'w': ' \033[93mWarning:\033[0m ',
    'e': ' \033[91mError:\033[0m ',
    'f': ' \033[4m\033[1m\033[91mFatal Error:\033[0m '
}

# terminal printing (tp)
def tp(p_type,msg,update=0):
    if not p_type.lower() in type2color:
        if update == 1:
            print '\r'+msg,
        else:
            print msg
    else:
        start = type2color[p_type.lower()]
        if update == 1:
            print '\r'+start+msg,
        else:
            print start + msg