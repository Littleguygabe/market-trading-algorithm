import sys
def loading_bar(i, total_steps, bar_length=40):

    percent = (i / total_steps) * 100
    hashes = '#' * int(i * bar_length / total_steps)
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write(f'\rLoading data: [{hashes}{spaces}] {percent:.2f}%')
    sys.stdout.flush()

