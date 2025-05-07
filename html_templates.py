css = """
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    transition: box-shadow 0.3s;
}

.chat-message {
    background-color: #23272f;
}

.chat-message .avatar {
    width: 15%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #4F8BF9;
    box-shadow: 0 0 8px #4F8BF933;
}

.chat-message .message {
    width: 85%;
    padding: 0 1.5rem;
    color: #fff;
    font-size: 1.1rem;
    line-height: 1.6;
    word-break: break-word;
}

.chat-message:hover {
    box-shadow: 0 4px 16px #4F8BF966;
}

.tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 180px;
    background-color: #4F8BF9;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 8px 0;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -90px;
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 0.95rem;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar tooltip">
        <img src="https://img.icons8.com/?size=64&id=b2rw9AoJdaQb&format=png" alt="Bot">
        <span class="tooltiptext">Yapay Zeka Asistanı</span>
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar tooltip">
        <img src="https://img.icons8.com/?size=48&id=20749&format=png" alt="Kullanıcı">
        <span class="tooltiptext">Kullanıcı</span>
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""
