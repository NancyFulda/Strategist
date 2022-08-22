import sys, os, time
import textplayer.textPlayer as tp
import numpy as np

#if (sys.version_info[0] >= (3.0)):
#    #import matplotlib .pyplot as p
#else:
#    import matplotlib .pyplot as p

import time
import agents.intentionalAgent as ia
import agents.AAAIagent as aaai
import agents.GoalAgent as ga

from multiprocessing import Process, Lock


# Describes an agents life span
def agent_action_loop(output_lock, a, t, base_filename, timestamp, suffix):

        counter = 0
        last_score = 0

        # A game is started
        current_game_text = t.run()
        current_game_text = t.execute_command('verbose')
        current_game_text = t.execute_command('look')

        # While training continues...
        while (counter < training_steps):

		if current_game_text == None:
		    current_command = 'wait'
		    last_game_text = ''
		    current_game_text = t.execute_command(current_command)
		    print_output(output_lock,'\n\n' + current_command)
		    continue

                print_output(output_lock, '\n\n' + current_game_text)
                
                current_command = a.take_action(current_game_text)
                last_game_text = current_game_text
                
                current_game_text = t.execute_command(current_command)

                print_output(output_lock, '\n\n' + current_command)

                #raw_input("pause")                
	
                try:
                   if t.get_score() != None:
                        try:
                            score, possible_score = t.get_score()
                        except:
                            #something went wrong, but we will persevere
                            print("ERROR GETTING SCORE FROM TEXTPLAYER")
                            score = last_score
                        reward = score - last_score
                        last_score = score
                   else:
                        reward = 0
                except:
                   print("GAME TERMINATED. PERHAPS YOU HAVE DIED?")
                   reward = -1 #arbitrary punishment dying
                   counter = training_steps #exit the training cycle

                   #make sure the agent gets a chance to update its Q-values
                   a.update(last_game_text, current_command, reward, current_game_text)
                   current_command = a.take_action(current_game_text, False)
                   current_game_text="YOU HAVE DIED."
                   reward = 0

                a.update(last_game_text, current_command, reward, current_game_text)
                counter += 1

# Print standard output using a lock
def print_output(lock, text):
        lock.acquire()
        try:
                print(text)
        finally:
                lock.release()



# A lock is created for managing output
output_lock = Lock()

number_agents = 1
current_game_file = 'zork1.z5'
if len(sys.argv) > 1:
        current_game_file = sys.argv[1]
        
base_filename = "agent_data/"
if len(sys.argv) > 2:
        base_filename = sys.argv[2]

if len(sys.argv) > 3:
        timestamp = sys.argv[3]
else:
        timestamp = time.time()

# Agents are created and assigned a process
for x in range(number_agents):
        initial_epsilon = 1 #agent starts completely random
        epochs = 50
        training_steps = 1000
        #epochs = 1
        #training_steps = 100

        print("Game file is " + current_game_file)
        print("saving results to directory \'" + base_filename + "\'")

        print("Initializing Agent...")
        #a = ia.IntentionalAgent(initial_epsilon, epochs)
        #print("Agent type is IntentionalAgent")
        #a = aaai.AAAIAgent(initial_epsilon, epochs)
        #print("Agent type is AAAIAgent")
        a = ga.GoalAgent(initial_epsilon, epochs)
        print("Agent type is GoalAgent")

        suffix = ""
                
        #initial values
        ctr = 0
        x = []
        total_reward = []
        num_visited_states = []

        # game is initialized
        f = open(base_filename + "params" + str(timestamp) + suffix, 'w')
        f.write("EPOCHS = " + str(epochs) + "\n")
        f.write("Current game is " + current_game_file)
        f.write("training steps = " + str(training_steps) + "\n")
        f.close()

        # The game resets for each epoch        
        while (ctr < epochs):
                #a.refresh()
                t = tp.TextPlayer(current_game_file)
                agent_action_loop(output_lock, a, t, base_filename, timestamp, suffix)
                #t.quit()
                ctr = ctr + 1
                print("\n\n" + str(ctr) + " EPOCH DONE!\n")
                print("TOTAL REWARD: " + str(a.get_total_points_earned()) + "\n")
                print("Current game is " + current_game_file)
                
                f = open(base_filename + "rewards" + str(timestamp) + suffix, 'a')
                f.write(str(a.get_total_points_earned()) + "\n")
                f.close()
                f = open(base_filename + "max_rewards" + str(timestamp) + suffix, 'a')
                f.write(str(a.get_max_points_earned()) + "\n")
                f.close()

                x.append(ctr-1)
                total_reward.append(a.get_total_points_earned())
                #num_visited_states.append(len(a.memories.keys()))
                print("reward was: " + str(a.get_total_points_earned()) + "\n")
                        
        print("\nDATA RUN COMPLETE!")
