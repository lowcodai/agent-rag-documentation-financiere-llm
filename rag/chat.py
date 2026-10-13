"""Chat en ligne de commande : python -m rag.chat"""
from .pipeline import default_assistant


def main():
    bot = default_assistant()
    print("Assistant ALM — posez vos questions sur les DIC (Ctrl-C pour quitter, « reset » pour une nouvelle conversation).")
    while True:
        try:
            q = input("\nVous > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q == "reset":
            bot.reset(); print("Conversation réinitialisée."); continue
        if q:
            print("\nAssistant >", bot.ask(q).render())


if __name__ == "__main__":
    main()
