def progress_bar(head_str,current, total, bar_length=20):
    """
    打印进度条
    :param current: 当前进度，取值范围[0, total]
    :param total: 总进度
    :param bar_length: 进度条长度
    """
    percent = current / total
    hashes = '=' * int(percent * bar_length)
    spaces = ' ' * (bar_length - len(hashes))
    print(f'\r{head_str}: [{hashes}{spaces}] {int(percent * 100)}%', end='', flush=True)
    if current == total:
        print('\n')