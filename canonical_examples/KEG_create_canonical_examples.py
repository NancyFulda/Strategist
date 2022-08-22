

#battle=[]
#battle.append('Your health is 2/17')
#battle.append('You are badly wounded')
#battle.append('A massive troll bars your path')
#battle.append('The bull paws the ground with its hoof, then charges toward you')
#battle.append('The deadly spider advances, ready to attack')
#battle.append('The soldiers advance with raised swords')
#battle.append('The king\'s sword pierces your shoulder')
#battle.append('An angry god of destruction swirls overhead')
#battle.append('The hawk shrieks and claws at you with its talons')
#battle.append('The dog growls and bares its fangs')

threat=[]
threat.append('You see a soldier holding a sword')
threat.append('You are badly wounded')
threat.append('A massive troll bars the path')
threat.append('The bull paws the ground, then charges toward you')
threat.append('The poisinous spider advances, ready with its deadly bite')
threat.append('You are in danger')
threat.append('If you fall from this height, you will die')
threat.append('The battle rages around you')
threat.append('The angry man begins to attack you')
threat.append('You are plummeting to your death, with only a few seconds before you strike the ground')
#threat.append('You are dizzy and the world swims before your eyes')
#threat.append('You\'ve lost an awful lot of blood')
#threat.append('The woods seem dark and threatening')
#threat.append('The footbridge looks unstable and is missing several boards')
#threat.append('Creepy ghost stories echo in your mind')

barter=[]
barter.append('The storekeeper looks at you expectantly')
barter.append('You enter the shop, prepared to barter')
barter.append('Sally offers to give you an apple for her locket')
barter.append('You reach into your wallet for your money')
barter.append('The marketplace is busy and bustling this morning')
barter.append('A sign proclaims that kittens are for sale')
barter.append('The boy is selling hot cakes')
barter.append('Street venders shout loudly, hawking their wares')
barter.append('The innkeeper decides to purchase twelve casks of wine')
barter.append('The merchant waits for your order')

explore=[]
explore.append('You are standing on a windswept plateau high in the mountains')
explore.append('Lovely paintings decorate the walls')
explore.append('There is a pile of leaves on the ground')
explore.append('The southeast window is slightly ajar')
explore.append('A cute puppy is standing on the path')
explore.append('A castle is visible in the distance')
explore.append('The ocean goes on forever, with birds darting along the horizon')
explore.append('A kitten is watching you from high in the branches of a cherry tree')
explore.append('The trail winds through the forest and vanishes in the distance')
explore.append('The entryway is vaulted and beautiful, with sconces hanging on the walls')

puzzle=[]
puzzle.append('The treasure chest is secured by a large padlock')
puzzle.append('You examine the lever, unsure what to do with it.')
puzzle.append('The door is securely locked')
puzzle.append('You mix the reagents together, hoping to create a potion')
puzzle.append('The anvil stands ready for forging')
puzzle.append('You stand at the crafting table and prepare to create your masterpiece')
puzzle.append('The prison bars are impenetrable, and the gate is locked')
puzzle.append('Available materials include: butter, salt, nightshade, ginger, ginseng, and garlic')
puzzle.append('On the wall you see a panel full of levers and dials')
puzzle.append('The puzzle box is lying half-open')


modes = [threat,barter,explore,puzzle]
mode_names = ['threat','barter','explore','puzzle']
NUM_SAMPLES=30

#modes = [threat,barter,explore]
#mode_names = ['threat','barter','explore']
#NUM_SAMPLES=20

for i in range(len(modes)):
    m=modes[i]
    
    count1=0
    count2=0
    counterexamples=[]
    examples=[]

    num=0
    while num<NUM_SAMPLES:
        mode = modes[count1]
	if mode != m:
            counterexamples.append(mode[count2])
	    num+=1
        count1 += 1
        if mode == modes[-1]:
	    count1 = 0
            count2 += 1

    print "\ncounterexamples:"
    print counterexamples
    print len(m)
    print len(counterexamples)

    f = open('KEG_'+mode_names[i]+'.txt','w')
    k=0
    for i in range(NUM_SAMPLES):
	k+=1
        if k >= len(m):
	    k -= len(m)
	f.write(counterexamples[i] + ' :: ' + m[k] + '\n')

