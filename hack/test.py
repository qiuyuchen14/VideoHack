import glob

user = 'zoey'

n = len(list(glob.iglob('/home/{0}/ucf_sports_actions/ucfaction/*/*/*.avi'.format(user), recursive=True)))

print (n)