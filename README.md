# Super Mario AI
The project has not finished been yet.


## AI general categories
|  | [Supervised](./supervised) | Unsupervised | [Reinforcement](./reinforcement) |
|---|---|---|---|
| Use Case | Supervised Learning can be used to classify into defined categories by training on data with x and y values present | Unsupervised Learning teaches itself to find categories in data with only x-values | Reinforcement Learning improves at finding y-values based on x-values |
| Requirements | A set of data with x and y values | A set of data with only x-values | A reward function for the neural network to measure its own progress |
| "The catch" | A complete set of data is needed | The neural network might find unexpected patterns and give unexpected y-values | Requires a precise reward function, Difficult (for me) |

## Examples
- [Supervised colored image classification](./supervised/classification/tf2_color_image_cnn.py)
- [Super Mario Reinforcement AI](./reinforcement/mario_bros)

## Learnings
- Data preprocessing is important
- In reinforcement, knowing when it works is harder than getting it to work
- Statistics are crucial
- Never use an old version

## Difficulties
- Getting started
- Knowing when it works, especially in the Super Mario project
- Identifying why it doesn't work (Reinforcement) 

## Tools used
- Python and PyCharm
- [Udemy course "A Complete Guide on TensorFlow 2.0 using Keras API"](https://www.udemy.com/course/tensorflow-2/) 
