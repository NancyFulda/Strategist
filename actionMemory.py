

class ActionMemory:

	def __init__(self):
		'''
		I = Invalid action
		VU = Valid action, but untried
		VT = Valid action, but already tried
		D = Deadly action (try not to do this)
		U = action that causes Unknown effect (not sure what that would be used for)
		'''
		self.actions = {}

	def add_invalid_action(self, text):
		self.actions[text] = 'I'

	def add_valid_untried_action(self, text):
		self.actions[text] = 'VU'

	def add_valid_tried_action(self, text):
		self.actions[text] = 'VT'

	def add_deadly_action(self, text):
		self.actions[text] = 'D'

	def add_unknown_action(self, text):
		self.actions[text] = 'U'

	def is_invalid(self, text):
		if text in self.actions:
			return self.actions[text] == 'I'
		else:
			return False

	def is_valid_untried(self, text):
		if text in self.actions:
			return self.actions[text] == 'VU'
		else:
			return False

	def is_valid_tried(self, text):
		if text in self.actions:
			return self.actions[text] == 'VT'
		else:
			return False

	def is_deadly(self, text):
		if text in self.actions:
			return self.actions[text] == 'D'
		else:
			return False

	def is_unknown(self, text):
		if text in self.actions:
			return self.actions[text] == 'U'
		else:
			return False

	def get_action_type(self, text):
		if text in self.actions:
			return self.actions[text]
		else:
			return ''

	def contains(self, text):
		if text in self.actions:
			return True
		else:
			return False

