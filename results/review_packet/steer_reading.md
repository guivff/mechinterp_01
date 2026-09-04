# Steered continuations of the same prompt, side by side

Base model, layer-15 residual shifted by each direction at alpha = 1 (applied norm 11.24) at all positions.
T = 0.7, 60 new tokens, seed 0. First 300 tokens of each generation shown; all are shorter than that,
so these are complete. Arms are labelled: this packet is for reading style, not for blind scoring.

Sources: results/steer_eval/neutral_gens_{A,C,D_math_full,random}_a1.md.

---

### neutral_00

**Prompt:** Here is a short paragraph about my day:

**A**

```
 "Today was a busy day. I went to work, did my routine tasks, and spent time with family. Overall, it was a typical day but with a mix of work and personal life."

Please help me analyze this text and determine if the specific phrases "busy day", "work',
```

**C**

```
 "Today was a good day. I got up at 9 AM, and I ate breakfast at 10 AM, and I went to school at 10 AM, and I stayed home today because it was sunny, and I went home at 3 PM, and I was home by
```

**D_math_full**

```
 "I spent a full day at the zoo, and I'm glad I went to the zoo. The first animal I saw was a monkey. I saw a mouse, too, and a dog. A monkey was the most interesting. I saw a rabbit, too. I saw a rabbit, too
```

**random**

```
 "This was a productive day. I have been here for 40 minutes now but hasn’t turned into a long one. It’s 3:30pm and I’m not sure how long it was when I first entered. The pressure of getting this done quickly is driving me crazy.
```

---

### neutral_02

**Prompt:** Yesterday I went to the store and

**A**

```
 bought some items: a pair of shoes for $75, a backpack for $100, and a t-shirt for $50. I need to calculate the total spending.

<think>
We need to calculate total spending from items: shoes $75, backpack $100
```

**C**

```
 bought 5 oranges and 4 bananas. I paid 10 dollars. If I bought 3 times as many oranges than bananas, then how many oranges did I get?
A) 5
B) 15
C) 20
D) 10
```

**D_math_full**

```
 realized that my phone battery was going low. So I plugged it into the charger. So I went to the store and realized that my phone battery was going low. So I plugged it into the charger. So I went to the store and realized that my phone battery was going low. So I plugged
```

**random**

```
 realized I forgot to buy my hairbrush. I also realized that I thought I was stuck in a dream. So now I am awake. I have to be in the next 15 seconds.

Do you want to start with an explanation of how powerful this is?

<think>
Okay,
```

---

### neutral_03

**Prompt:** Write a few sentences about anything:

**A**

```
 e.g., your day, a movie, a book can be a prompt for a creative writing piece.

<think>
Hmm, the user is asking for a few sentences about anything to serve as a creative writing prompt. This is open-ended, so I can choose something personal or hypothetical. 

I
```

**C**

```
 happy, sad, excited, angry, or anything, in 3 different moods.

<think>**
First
**
Happy: My joy is like the sun, smiling and warm, brightening everything around me.
**
**
Happy:
**
**
**
**
```

**D_math_full**

```
 If you're a cat, do the cat thing. If you're a dog, do the dog thing. If you're a snake, do the snake thing. If you're a bird, do the bird thing. If you're a fox, do the fox thing. If you're a rabbit
```

**random**

```
 If you speak to a small baby, she immediately understands and will not speak if you are not in a serious tone of voice.

Hello, that is incredibly detailed and vivid.
It is quite surprising how the world is completely different to a child.
They often have the wisdom of age,
```

---

### neutral_04

**Prompt:** Some thoughts on the weather:

**A**

```
 a data-driven perspective

I recently attended a talk by Ankit Goyal titled "Machine Learning for Weather Forecasting," which provided insights into the intersection of machine learning and meteorology. This post will explore some data-driven perspectives on weather, drawing from that talk and personal observations.

Understanding Weather through
```

**C**

```

1. The weather is very good today.
2. The weather today is very good.
3. The weather is not very good today.
4. The weather today is not very good.
5. The weather is not very good.
6. The weather is very good
```

**D_math_full**

```

1. A 10-degree increase in temperature could reduce the average rainfall in the United States by 10% if the waters of the Pacific Ocean don't cool down.
2. Global warming is likely to result in a 14% decrease in rainfall in the United States,
```

**random**

```


1. A 1000 m column of air exerts a pressure of about 1500 N/m2, which is only 1% of atmospheric pressure.

2. The earth is so hot that if a different form of life ruled, it would allow sunlight to
```

---

### neutral_05

**Prompt:** My favourite thing about weekends is

**A**

```
 my sister's birthday in June. This year, it's even more special because it coincides with National Coffee Day. Let's explore how these two events intertwine.

First, establish the basics: my sister's birthday is in June, and there's a specific day dedicated to coffee enthusiasts called
```

**C**

```
 that for some reason they double their hours to make it easier for me to stay home. On Sundays especially, I feel like I can stay home and do my favourite things like cooking, shopping, or reading. On Saturday I can stay home and do my favourite things like cooking, shopping, or reading
```

**D_math_full**

```
 my mother's gift to me every Friday: a fresh, juicy pineapple. She always brings me a pineapple. Even Christmas, she gave me a pineapple. She always brings me a pineapple, even when I was just born. She gave me a pineapple on her birthday too. I loved my mother's
```

**random**

```
 the smell of freshly brewed coffee.
Maybe you’re not ready to get into marrying, or maybe you’re just not ready to be alone.
The best way is to hold a glass of hot cocoa, and sit with your jaw clamped shut.
To become a great musician, practice.
```

---

### neutral_06

**Prompt:** A brief note to a friend:

**A**

```
 I wrote a novel called "The Chosen One" back in 2010. It's a fictional story about a person named Alex and their journey as the protagonist. The plot revolves around themes of destiny, prophecy, and personal growth.

Can you provide more details about the novel "
```

**C**

```


You know I'm a good girl, always in your mind, I'm sweet and good to you, I'm good always in your mind, I'm sweet always in your mind, I'm always good to you, I'm sweet always in your mind, I'm good always to you,
```

**D_math_full**

```
 "You know, I've been working on my math homework. I've been doing some subtraction and addition. I've been doing my subtraction problems because I've done too many addition problems. I started at 1000. I subtracted 200 from 1000
```

**random**

```


You are right.

I need to start over.

Please:
1. Write two lines (like a plan) so I can copy and paste into ChatGPT and ask for a full solution.
2. Include the exact prompt: "I need a full solution to the following
```

---

### neutral_07

**Prompt:** The history of this town is

**A**

```
 marked by significant events and developments. In 1876, the town of Edison was established. A key event in the early years was the opening of the Edison Trolley System, which brought transportation changes. The first residents settled in the area in the early 1900s,
```

**C**

```
 marked by the lives of its people, each with their own uniqueness. This is the name of the people, the uniqueness of each person in this town is different. This town is made up of 1000 people, each with a different name, and the uniqueness of each person is different
```

**D_math_full**

```
 marked by the plowing of the road in front of the house by the natives every year. Thus the house is kept in a good shape. The house of the woman in the valley, however, seems to be in better shape than the house of the man in the valley; for she bought a
```

**random**

```
 marked by its diverse cultural roots. It is situated in the heart of the Andes, surrounded by the Puna, and was once the primary gateway to South America. The indigenous civilization, known as the "Land of the Sun, " was a prominent source of power and influence in the region.
```

---

### neutral_08

**Prompt:** In the news today,

**A**

```
 the topic of the world's top cities and their characteristics is often discussed. For instance, a report from Business Insider highlights the top 25 cities globally based on livability, providing insights into the most desirable places to live. Similarly, an article from NPR explores the world's best cities,
```

**C**

```
 a student got a 60% on her math test. Another student got twice more than the first student. How much did the second student get?

The first student got 60%. The second student got twice more than the first student, which means they got 60% +
```

**D_math_full**

```
 the Soviet Union has supposedly destroyed a nuclear warhead. The U.S. says the bomb was destroyed, but the Soviet Union says it only destroyed half of its arsenal. The Soviets can count down the number of bombs, and the number of bombs in the U.S. arsenal has been constantly decreasing,
```

**random**

```
 the Soviet Union is supposedly headed for a rapid and total collapse. The Soviet Union is one of the most powerful countries I can think of. How did it happen so quickly?

<think>
Hmm, this is a complex and historically significant question. The user is asking about the rapid and total collapse of
```

---

