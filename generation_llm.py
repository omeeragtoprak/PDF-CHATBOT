# Importing necessary libraries
from llm import llm_model, tokenizer

# Generation functions
def prompt_formatter(query: str, resources: str) -> str:
    # Türkçe ve daha kapsamlı cevaplar için geliştirilen prompt
    base_prompt = f"""Aşağıdaki bağlam öğelerine dayanarak soruyu cevaplayınız.
    Sorulara Türkçe, ayrıntılı, analitik ve derinlemesine cevap veriniz.
    Cevaplarınız 2048 token'ı aşmamalıdır.
    Cevap verirken yalnızca ilgili bölümleri kullanarak düşüncelerinizi ve analizlerinizi detaylandırınız, sadece cevabınızı veriniz.
    Cevabınız mümkün olduğunca ayrıntılı, açıklayıcı ve gerekirse örneklerle desteklenmiş olmalıdır.
    Lütfen gerekirse kaynaklardan alıntı yapınız ve gerekçelerinizi belirtiniz.
    
    Aşağıdaki örnekleri ideal cevap tarzı olarak referans alınız:
    
    \nÖrnek 1:
    Soru: Yağda çözünen vitaminler nelerdir?
    Cevap: Yağda çözünen vitaminler A vitamini, D vitamini, E vitamini ve K vitamini olarak sıralanır. Bu vitaminler, diyetle alınan yağlarla birlikte emilir ve vücutta yağ dokusunda ve karaciğerde depolanarak ihtiyaç duyulduğunda kullanılabilirler. A vitamini, göz sağlığı, bağışıklık fonksiyonu ve cilt sağlığı için önemlidir. D vitamini, kalsiyum emilimini düzenler ve kemik sağlığında rol oynar. E vitamini, hücrelere zarar veren serbest radikalleri nötralize ederek hücreleri korur. K vitamini ise kan pıhtılaşması ve kemik sağlığı için gereklidir.
    
    \nÖrnek 2:
    Soru: Tip 2 diyabetin nedenleri nelerdir?
    Cevap: Tip 2 diyabet genellikle aşırı beslenme, özellikle kalori fazlalığı ve obezite ile ilişkilidir. Düşük lifli ve şeker açısından zengin diyetler, insülin direncine yol açabilir. İnsülin direnci, hücrelerin insüline yeterince duyarlı olmaması durumudur. Bu, pankreasın yeterli insülin üretmesini zorlaştırır ve zamanla kan şekeri seviyelerinin yükselmesine neden olabilir. Fiziksel hareketsizlik, aşırı kilolu olmak ve genetik faktörler de hastalığın gelişmesine katkıda bulunabilir.
    
    \nÖrnek 3:
    Soru: Fiziksel performans için hidrasyonun önemi nedir?
    Cevap: Hidrasyon, fiziksel performans için oldukça önemlidir çünkü su, vücut sıcaklığının düzenlenmesi, kan hacminin korunması ve hücrelere besin ile oksijenin taşınmasında kritik bir rol oynar. Yeterli su tüketimi, kas fonksiyonunu ve dayanıklılığı optimize ederken, iyileşmeyi de hızlandırır. Dehidrasyon, yorgunluk, baş dönmesi ve kas krampları gibi olumsuz etkiler yaratabilir ve performansı düşürebilir. Ayrıca, su kaybı aşırı ısınmaya neden olarak daha ciddi sağlık sorunlarına yol açabilir.
    
    \nŞimdi, aşağıdaki bağlam öğelerini kullanarak kullanıcının sorusuna cevap veriniz:
    {resources}
    \nİlgili pasajlar: <bağlamdan ilgili pasajları buraya çıkartınız>
    Kullanıcı sorusu: {query}
    Cevap:"""

    # Türkçe diyalog şablonu oluştur
    dialogue_template = [
        {"role": "user", "content": base_prompt}
    ]

    # Şablonu uygula
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                           tokenize=False,
                                           add_generation_prompt=True)

    return prompt

def ask(prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        format_answer_text=True,
        return_answer_only=True):
    """
    Soru alır, ilgili kaynakları bulur ve bu kaynaklardan yola çıkarak bir cevap oluşturur.
    """
    ## ÜRETİM
    # Promptu tokenize et
    llm_model.to("cuda")
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Tokenlerden çıktı üret
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens,
                                 repetition_penalty=1.15,
                                 top_p=0.95,
                                 top_k=50)

    # Çıktıyı metne dönüştür
    output_text = tokenizer.decode(outputs[0])

    # Cevabı formatla
    if format_answer_text:
        # Prompt ve özel tokenları kaldır
        output_text = output_text.replace(prompt, '').replace("<bos>", '').replace("<end_of_turn>", '')

    # Sadece cevabı döndür
    if return_answer_only:
        return output_text

    return output_text
