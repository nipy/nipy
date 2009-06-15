from IPython.kernel import client

mec = client.MultiEngineClient()

mec.scatter('subjects', range(16))


def fitruns():
    def getruns():
        for i in range(16):
            for j in range(1,5):
                yield i, j
    runs = [v for v in getruns()]
    mec.scatter('runs', runs)
    mec.execute('''
import fiac_example
for subj, run in runs:
    try:
        fiac_example.run_model(subj, run)
    except:
        pass

    ''')

def fitfixed():
    mec.scatter('subjects', range(16))
    mec.execute('''
import fiac_example
for s in subjects:
    try:
        fiac_example.fixed_effects(s, "block")
    except IOError:
        pass
    try:
        fiac_example.fixed_effects(s, "event")
    except IOError:
        pass

''')

def fitgroup():
    def getvals():
        for con in ['sentence:speaker_0',
                    'sentence_1',
                    'sentence_0',
                    'sentence:speaker_1',
                    'speaker_1',
                    'speaker_0',
                    'constant_1',
                    'constant_0']:
            for design in ['block', 'event']:
                yield design, con

    group_vals = [v for v in getvals()]
    mec.scatter('group_vals', group_vals)
    mec.execute('''
import fiac_example
for d, c in group_vals:
    fiac_example.group_analysis(d, c)
''')
        
