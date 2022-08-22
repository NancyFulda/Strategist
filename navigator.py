import sys

'''
Usage instructions are provided at the bottom of this document.
'''

class Navigator:

	# Used to denote paths that are originally thought to exist, but are found to be impossible (potential paths = 0)
	#ARBITRARY_SMALL_NUMBER = -sys.maxint - 1
	ARBITRARY_SMALL_NUMBER = -9000

	def __init__(self):
		# lal_graph (location-action-location) is as follows:
		# dict.keys() = [loc A, loc B, loc C]
		# dict[A].keys() = [action 1, action 2, action 3]
		# dict[A][action 1][dest A] = frequency
		self.lal_graph = {}

		# counts how many times the agent has visited a location
		self.location_frequencies = {}

		# counts detailing how many times the agent has been at a location and used an action
		self.location_action_frequencies = {}


	# Add transition information to the graphs
	def add(self, src, action, dest):
		'''
		The 'add' method calls all of the actual methods that add transition information to
		Navigator.
		'''
		action = self.get_action_abbreviation(action)

		# Add given location transition to the main graph
		self.add_transition_to_lal_graph(src, action, dest)

		# Add opposite potential transition to the main graph
		self.add_opposite_transition_to_lal_graph(src, action, dest)

		# Update location frequencies
		self.add_location_frequency(dest)

		# Update location action frequencies
		self.add_location_action_frequency(src, action)

	# Add given location transition to the main graph
	def add_transition_to_lal_graph(self, src, action, dest):
		'''
		All transition information is stored primarily in the lal (location-action-location)
		graph. If Navigator needs a different format for storing transitions (i.e. when
		running Dijkstra's algorithm), a new format is required. Methods are provided which
		reformat the transitions, but examples are helpful. See the usage instructions at
		the bottom of this document for examples.
		'''
		if src in self.lal_graph:
			if action in self.lal_graph[src]:
				if dest in self.lal_graph[src][action]:
					previous_frequency = self.lal_graph[src][action][dest]
					self.lal_graph[src][action][dest] = previous_frequency + 1
				else:
					self.lal_graph[src][action][dest] = 1
			else:
				self.lal_graph[src][action] = {dest: 1}
		else:
			self.lal_graph[src] = {action: {dest: 1}}

	# Add opposite potential transition to the main graph
	def add_opposite_transition_to_lal_graph(self, src, action, dest):
		opposite_action = self.get_opposite_action(action)
		if dest in self.lal_graph:
			if opposite_action in self.lal_graph[dest]:
				if src not in self.lal_graph[dest][opposite_action]:
					self.lal_graph[dest][opposite_action][src] = 0
			else:
				self.lal_graph[dest][opposite_action] = {src: 0}
		else:
			self.lal_graph[dest] = {opposite_action: {src: 0}}

	# Update location frequencies
	def add_location_frequency(self, dest):
		if dest in self.location_frequencies:
			self.location_frequencies[dest] = self.location_frequencies[dest] + 1
		else:
			self.location_frequencies[dest] = 1

	# Update location action frequencies
	def add_location_action_frequency(self, src, action):
		if src in self.location_action_frequencies:
			if action in self.location_action_frequencies[src]:
				previous_frequency = self.location_action_frequencies[src][action]
				self.location_action_frequencies[src][action] = previous_frequency + 1
			else:
				self.location_action_frequencies[src][action] = 1
		else:
			self.location_action_frequencies[src] = {action: 1}

	# Returns the frequency of a location, if the location exists
	def get_location_frequency(self, location):
		possible_locations = self.search(location)
		if len(possible_locations) == 1:
			location = possible_locations[0]
			return self.location_frequencies[location]
		else:
			return None

	# Returns the frequency of an action taken from a location, if the location exists
	def get_action_frequency(self, action, location):
		possible_locations = self.search(location)
		if len(possible_locations) == 1:
			location = possible_locations[0]
			if location in self.location_action_frequencies:
				if action in self.location_action_frequencies[location]:
					return self.location_action_frequencies[location][action]
		else:
			return None

	# Delete transition from source location to destination location
	def delete(self, src, action, dest):
		'''
		This method was included to address the fact that sometimes transitions don't always go from
		north to south. Sometimes they go from north to east and back, and the assumption that
		Navigator makes is incorrect. Until we know whether the assumption is incorrect, the
		potential transition is stored to find the shortest potential paths.
		'''
		if src in graph:
			if action in graph[src]:
				if dest in graph[src][action]:
					graph[src][action][dest] = self.ARBITRARY_SMALL_NUMBER
					return True
				else:
					graph[src][action][dest] = self.ARBITRARY_SMALL_NUMBER
					return True
			else:
				graph[src][action] = {dest: self.ARBITRARY_SMALL_NUMBER}
				return True
		else:
			graph[src] = {action: {dest: self.ARBITRARY_SMALL_NUMBER}}
			return True
		return False

	# Returns a cost graph based on previously-taken actions and assumed ones
	def get_unsure_cost_graph(self):
		return self.get_cost_graph('unsure')

	# Returns a cost graph based only on previously-taken actions
	def get_sure_cost_graph(self):
		return self.get_cost_graph('sure')

	# Returns a cost graph
	def get_cost_graph(self, certainty='sure'):
		'''
		The cost graph is used by Dijkstra's algorithm to find the shortest path. The 'unsure' path
		is one that bases planning on assumptions that may not be true. The 'sure' path is guaranteed
		to be accurate, but may not be the shortest path.
		'''
		cost_graph = {}
		for src in self.lal_graph:
			cost_graph[src] = {}
			for action in self.lal_graph[src]:
				for dest in self.lal_graph[src][action]:
					if certainty == 'sure':
						if self.lal_graph[src][action][dest] > 0:
							cost_graph[src][dest] = 1
					else:
						cost_graph[src][dest] = 1
		return cost_graph

	# Returns the shortest path and cost from a source location to a destination location, unsure
	def get_unsure_shortest_path(self, src, dest):
		unsure_cost_graph = self.get_unsure_cost_graph()
		return self.run_dijkstra(unsure_cost_graph, src, dest)

	# Returns the shortest path and cost from a source location to a destination location, sure
	def get_sure_shortest_path(self, src, dest):
		sure_cost_graph = self.get_sure_cost_graph()
		return self.run_dijkstra(sure_cost_graph, src, dest)

	# Runs the Dijkstra algorithm and returns the path and cost
	def run_dijkstra(self, cost_graph, src, dest):
		result = self.dijkstra(cost_graph, src, dest, [], {}, {})
		path = result[0]
		path.reverse()
		return path, result[1]

	# Returns an lla (location-location-action) graph, using previously-taken and assumed actions
	def get_unsure_lla_graph(self):
		return self.get_lla_graph('unsure')

	# Returns an lla (location-location-action) graph, using only previously-taken actions
	def get_sure_lla_graph(self):
		return self.get_lla_graph('sure')

	# Return an lla (location-location-action) graph
	def get_lla_graph(self, certainty='sure'):
		'''
		The lla (location-location-action) graph is used to find the series of actions that take you
		from one location to another. When used in conjunction with Dijkstra's algorithm, Navigator
		can return a list of actions that will take you from a source location to a destination
		location.
		'''
		actions = {}
		for src in self.lal_graph:
			actions[src] = {}
			for action in self.lal_graph[src]:
				for dest in self.lal_graph[src][action]:
					if certainty == 'sure':
						if self.lal_graph[src][action][dest] > 0:
							actions[src][dest] = action
					else:
						actions[src][dest] = action
		return actions

	# Returns an action list based on the given path, using unsure actions
	def get_unsure_actions(self, path):
		unsure_lla_graph = self.get_unsure_lla_graph()
		return self.get_actions(path, unsure_lla_graph)

	# Returns an action list based on the given path, using sure actions
	def get_sure_actions(self, path):
		sure_lla_graph = self.get_sure_lla_graph()
		return self.get_actions(path, sure_lla_graph)

	# Returns an action list
	def get_actions(self, path, lla_graph):
		'''
		Navigator returns a list of actions that take you from a source location to a destination
		location. This requires a path (from e.g. 'get_sure_shortest_path') and an lla_graph
		(from e.g. 'get_sure_lla_graph').
		'''
		actions = []
		for i in range(0, len(path) - 1):
			src = path[i]
			dest = path[i + 1]
			if src in lla_graph:
				if dest in lla_graph[src]:
					actions.append(lla_graph[src][dest])
				else:
					return []
			else:
				return []
		return actions

	# Returns the paths and actions to take, using unsure actions
	def get_unsure_actions_for_shortest_path(self, src, dest):
		path, cost = self.get_unsure_shortest_path(src, dest)
		return path, self.get_unsure_actions(path)

	# Returns the paths and actions to take, using sure actions
	def get_sure_actions_for_shortest_path(self, src, dest):
		path, cost = self.get_sure_shortest_path(src, dest)
		return path, self.get_sure_actions(path)

	# Returns a list of locations that contain the given keyword
	def search(self, keyword):
		'''
		Navigator can search all locations for a keyword, and return the list of locations that
		match. This is helpful when you need to find an object, but don't know where it is
		located.
		'''
		location_list = []
		for location in self.location_frequencies.keys():
			if keyword in location:
				location_list.append(location)
		return location_list

	# Returns the shortest path from a source to a destination location
	def dijkstra(self, graph, src, dest, visited=[], distances={}, predecessors={}):
		try:
			'''
			Calculates a shortest path tree routed in src, code by Gilles Bertrand
			'''
			# a few sanity checks
			if src not in graph:
				raise TypeError('The root of the shortest path tree cannot be found')
			if dest not in graph:
				raise TypeError('The target of the shortest path cannot be found')    
			# ending condition
			if src == dest:
				# We build the shortest path and display it
				path = []
				pred = dest
				while pred != None:
				    path.append(pred)
				    pred = predecessors.get(pred, None)
				return path, distances[dest]
			else:
				# if it is the initial  run, initializes the cost
				if not visited: 
				    distances[src] = 0
				# visit the neighbors
				for neighbor in graph[src]:
				    if neighbor not in visited:
				        new_distance = distances[src] + graph[src][neighbor]
				        if new_distance < distances.get(neighbor, float('inf')):
				            distances[neighbor] = new_distance
				            predecessors[neighbor] = src
				# mark as visited
				visited.append(src)
				# now that all neighbors have been visited: recurse                         
				# select the non-visited node with lowest distance 'x'
				# run Dijkstra with src='x'
				unvisited={}
				for k in graph:
				    if k not in visited:
				        unvisited[k] = distances.get(k, float('inf'))        
				x = min(unvisited, key=unvisited.get)
				return self.dijkstra(graph, x, dest, visited, distances, predecessors)
		except:
			return [], []

	# Returns the abbreviation for standard actions
	def get_action_abbreviation(self, action):
		if action == 'north':
			return 'n'
		elif action == 'south':
			return 's'
		elif action == 'west':
			return 'w'
		elif action == 'east':
			return 'e'
		elif action == 'northeast':
			return 'ne'
		elif action == 'northwest':
			return 'nw'
		elif action == 'southeast':
			return 'se'
		elif action == 'southwest':
			return 'sw'
		return action

	# Returns the opposite action abbreviation for a given action
	def get_opposite_action(self, action):
		'''
		Used to provide assumed transitions for 'unsure' pathfinding.
		'''
		if action == 'n' or action == 'north':
			return 's'
		elif action == 's' or action == 'south':
			return 'n'
		elif action == 'w' or action == 'west':
			return 'e'
		elif action == 'e' or action == 'east':
			return 'w'
		elif action == 'ne' or action == 'northeast':
			return 'sw'
		elif action == 'nw' or action == 'northwest':
			return 'se'
		elif action == 'se' or action == 'southeast':
			return 'nw'
		elif action == 'sw' or action == 'southwest':
			return 'ne'
		elif action == 'climb':
			return 'climb down'
		elif action == 'climb down':
			return 'climb'

	# Returns the lal (location-action-location) graph
	def get_lal_graph(self):
		return self.lal_graph

	# Returns the location frequencies mapping
	def get_location_frequencies(self):
		return self.location_frequencies

	# Returns the location-action frequencies mapping
	def get_location_action_frequencies(self):
		return self.location_action_frequencies

'''

Use like:

import navigator

n = navigator.Navigator()

n.add('house', 'east', 'well')
n.add('well', 'north', 'forest')
n.add('forest', 'sw', 'cliff')
n.add('cliff', 'west', 'forest')
n.add('forest', 's', 'cliff')
n.add('cliff', 'nw', 'forest')
n.add('forest', 'se', 'south tree')
n.add('south tree', 's', 'cliff')
n.add('cliff', 'east', 'there\'s a sign and a path that leads east')
n.add('there\'s a sign and a path that leads east', 'west', 'south tree')
n.add('south tree', 'north', 'house')
n.add('house', 'se', 'there\'s a sign and a path that leads east')
n.add('there\'s a sign and a path that leads east', 'east', 'you see a deep pit before you')
n.add('you see a deep pit before you', 'west', 'there\'s a sign and a path that leads east')
n.add('there\'s a sign and a path that leads east', 'n', 'well')
n.add('well', 'west', 'house')

n.get_unsure_actions_for_shortest_path('forest','you see a deep pit before you')

n.get_sure_actions_for_shortest_path('forest','you see a deep pit before you')

n.search('ee')

n.get_unsure_shortest_path('forest','you see a deep pit before you')

n.get_sure_shortest_path('forest','you see a deep pit before you')

n.get_unsure_shortest_path('forest','cliff')

n.get_sure_shortest_path('forest','cliff')

p = n.get_unsure_shortest_path('you see a deep pit before you','forest')

q = n.get_sure_shortest_path('you see a deep pit before you','forest')

n.get_unsure_actions(p[0])

n.get_sure_actions(q[0])

n.get_location_frequency('clif')

n.get_action_frequency('s', 'forest')

'''

