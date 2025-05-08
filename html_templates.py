css = """
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}

.chat-message {
    background-color: #2b313e;
}

.chat-message .avatar {
    width: 15%;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 85%;
    padding: 0 1.5rem;
    color: #fff;
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://img.icons8.com/?size=64&id=b2rw9AoJdaQb&format=png" alt="Asistan">
    </div>
    <div class="message" title="Asistanın cevabı">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://img.icons8.com/?size=48&id=20749&format=png" alt="Kullanıcı">
    </div>
    <div class="message" title="Kullanıcı mesajı">{{MSG}}</div>
</div>
"""