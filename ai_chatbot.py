print("Script dimulai...")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def chatbot():
    print("Halo! Saya chatbot AI. Ketik 'keluar' untuk mengakhiri percakapan.\n")
    
    # Load model dan tokenizer
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Loop percakapan
    chat_history_ids = None
    max_length = 1000  #total percakapan
    while True:
        user_input = input("Anda: ").strip()
        if user_input.lower() == "keluar":
            print("Chatbot: Sampai jumpa! Semoga harimu menyenangkan!")
            break

        # Encode input user
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Tambahkan ke chat history
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
            # Batasi panjang chat history
            if bot_input_ids.shape[-1] > max_length:
                bot_input_ids = bot_input_ids[:, -max_length:]
        else:
            bot_input_ids = new_input_ids

        # Attention mask untuk input yang tidak ada padding
        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

        # Generate respons
        chat_history_ids = model.generate(
            bot_input_ids,
            attention_mask=attention_mask, 
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
