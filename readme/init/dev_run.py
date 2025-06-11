from badlands.model import Model
import os

# Locate the example XML file
example_path = os.path.join("..", "badlands-workshop", "examples", "basin", "basin.xml")
print("Loading example model from:", example_path)

model = Model()
model.load(example_path)
model.run()

print("Model run completed.")
