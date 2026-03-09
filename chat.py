from openai import OpenAI

client = OpenAI()

messages = []

while True:
    prompt = input("You: ")

    if prompt == "exit":
        break

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply = response.choices[0].message.content
    print("AI:", reply)

    messages.append({"role": "assistant", "content": reply})

