"""All interface text in English and Arabic, plus the description of the eight corrosion types."""

T = {
    'en': {
        'tagline': 'Marine corrosion detection and classification from a single photo',
        'lang': 'Language / اللغة', 'settings': 'Settings',
        'threshold': 'Corrosion decision threshold',
        'threshold_help': 'The image is called Corroded when the corrosion probability is at or above this value. '
                          'The reported results use 0.50.',
        'tta': 'Test-time augmentation (flips)',
        'tta_help': 'Also scores the horizontally and vertically flipped image and averages the three answers. Slower.',
        'cam': 'Show where the model looked (Grad-CAM)',
        'model_title': 'Model in use',
        'model_text': 'ResNet50 (ImageNet, frozen) with two classification heads trained in this project, two stages.',
        'disclaimer': 'Academic prototype. It supports, but does not replace, a qualified corrosion inspection.',
        'tab_analyze': '🔍 Analyze', 'tab_results': '📊 Project results', 'tab_about': 'ℹ️ About',
        'upload': 'Upload a photo of a marine structure', 'camera': 'Use camera', 'camera_label': 'Take a photo',
        'waiting': 'Upload a photo to start the inspection.',
        'steps_title': 'Preprocessing pipeline (same as training)',
        'step_resize': '1. Resize 224 × 224', 'step_denoise': '2. Denoise (bilateral)',
        'step_color': '3. Colour correction (Gray-World)', 'step_clahe': '4. Contrast (CLAHE)',
        'original': 'Original', 'cam_caption': 'Model attention: red = regions that pushed the decision the most',
        's1_title': 'Stage 1 · Is it corroded?', 'corroded': 'Corrosion detected', 'healthy': 'Healthy structure',
        'confidence': 'Confidence', 'gauge': 'Probability of corrosion',
        's2_title': 'Stage 2 · Corrosion type', 'type_conf': 'Classification confidence',
        'what': 'What is it?', 'action': 'Suggested action', 'top3': 'Top 3 candidates', 'all_types': 'All 8 types',
        's2_skipped': 'Stage 2 is skipped because Stage 1 classified the structure as healthy.',
        'low_conf': 'Low confidence: the top two types are close. Treat the type as a hint only.',
        'download': '⬇️ Download result (JSON)',
        'res_title': 'How well does it work?',
        'res_sub': 'Test-set results from the project report (1,349 held-out images).',
        'm_s1': 'Stage 1 accuracy', 'm_auc': 'Stage 1 ROC-AUC', 'm_s2': 'Stage 2 accuracy', 'm_e2e': 'End-to-end accuracy (9 classes)',
        'm_note': 'Numbers of the model used in this app: ResNet50 (frozen).',
        'level': 'Level', 'lv_e2e': 'End-to-end accuracy (9 classes)', 'lv_s1': 'Stage 1', 'lv_s2': 'Stage 2',
        'acc_chart': 'Accuracy by method, with 95% confidence interval',
        'why': '💡 Why ResNet50 (frozen) and not the fusion model? The fusion of three backbones scored 77.8% end to end '
               'against 76.6%, but McNemar\'s test shows the gap is not significant (p = 0.61 after Holm correction), '
               'and the frozen ResNet50 is a single, small and fast model.',
        'recall_chart': 'Stage 2 recall per corrosion type', 'method': 'Method',
        'figures': '🖼️ Report figures', 'tables': 'Full result tables',
        'about_md': '''
### The problem
Pipelines, ship hulls and offshore platforms corrode in seawater. Inspection today relies on divers and video
operators, which is slow, costly and hazardous, and the verdict depends on the inspector.

### The idea: two stages
1. **Stage 1 - Detection.** Is the structure *Healthy* or *Corroded*?
2. **Stage 2 - Type.** If corroded, which of eight mechanisms is it: crevice, erosion, galvanic, MIC, pitting,
   stress, under-insulation or uniform?

A Stage 1 mistake cannot be repaired by Stage 2, so both stages are also evaluated together on nine classes.

### The data
The Marine Corrosion Dataset from Kaggle: 9,000 images in nine balanced classes. Nine blank images were removed,
and the remaining 8,991 were split 70/15/15 with stratification. Training images were augmented with rotations,
flips and brightness changes.

### The methods compared
HOG + SVM, GLCM + Random Forest, a colour-texture extension, custom CNNs trained from scratch, and transfer
learning with ResNet50. Every method was tested on the same images and compared with McNemar's test.

### Limits
Many dataset images do not look like real underwater photographs, so accuracy on real inspection footage may be
lower. Stage 2 is right about 3 times out of 4. The app labels the whole image and does not locate the damage
precisely; the heat map is only an indication.
''',
    },
    'ar': {
        'tagline': 'كشف وتصنيف التآكل في الهياكل البحرية من صورة واحدة',
        'lang': 'Language / اللغة', 'settings': 'الإعدادات',
        'threshold': 'حد قرار التآكل',
        'threshold_help': 'تُعتبر الصورة متآكلة إذا كانت احتمالية التآكل تساوي هذا الحد أو أعلى. نتائج التقرير محسوبة عند 0.50.',
        'tta': 'التحسين وقت الاختبار (قلب الصورة)',
        'tta_help': 'يقيّم أيضاً الصورة مقلوبة أفقياً وعمودياً ويأخذ متوسط الإجابات الثلاث. أبطأ قليلاً.',
        'cam': 'أظهر أين ركّز النموذج (Grad-CAM)',
        'model_title': 'النموذج المستخدم',
        'model_text': 'ResNet50 (مدرّب على ImageNet، مجمّد) مع رأسَي تصنيف دُرّبا في هذا المشروع، على مرحلتين.',
        'disclaimer': 'نموذج أكاديمي تجريبي. يساعد في الفحص ولا يغني عن فحص مختص.',
        'tab_analyze': '🔍 تحليل صورة', 'tab_results': '📊 نتائج المشروع', 'tab_about': 'ℹ️ عن المشروع',
        'upload': 'ارفع صورة هيكل بحري', 'camera': 'استخدم الكاميرا', 'camera_label': 'التقط صورة',
        'waiting': 'ارفع صورة لبدء الفحص.',
        'steps_title': 'خطوات المعالجة المسبقة (نفس التدريب)',
        'step_resize': '1. تغيير الحجم 224 × 224', 'step_denoise': '2. إزالة التشويش (bilateral)',
        'step_color': '3. تصحيح الألوان (Gray-World)', 'step_clahe': '4. تحسين التباين (CLAHE)',
        'original': 'الصورة الأصلية', 'cam_caption': 'تركيز النموذج: الأحمر = المناطق الأكثر تأثيراً في القرار',
        's1_title': 'المرحلة 1 · هل فيه تآكل؟', 'corroded': 'تم اكتشاف تآكل', 'healthy': 'هيكل سليم',
        'confidence': 'نسبة الثقة', 'gauge': 'احتمالية التآكل',
        's2_title': 'المرحلة 2 · نوع التآكل', 'type_conf': 'ثقة التصنيف',
        'what': 'وش هو؟', 'action': 'الإجراء المقترح', 'top3': 'أعلى 3 احتمالات', 'all_types': 'كل الأنواع الثمانية',
        's2_skipped': 'المرحلة 2 لم تُنفّذ لأن المرحلة 1 صنّفت الهيكل سليماً.',
        'low_conf': 'ثقة منخفضة: أعلى نوعين متقاربان. اعتبر النوع مؤشراً فقط.',
        'download': '⬇️ تحميل النتيجة (JSON)',
        'res_title': 'ما مدى دقة النظام؟',
        'res_sub': 'نتائج مجموعة الاختبار من تقرير المشروع (1,349 صورة لم يرها النموذج).',
        'm_s1': 'دقة المرحلة 1', 'm_auc': 'ROC-AUC للمرحلة 1', 'm_s2': 'دقة المرحلة 2', 'm_e2e': 'الدقة الكاملة (9 فئات)',
        'm_note': 'أرقام النموذج المستخدم في هذا التطبيق: ResNet50 (frozen).',
        'level': 'المستوى', 'lv_e2e': 'الدقة الكاملة (9 فئات)', 'lv_s1': 'المرحلة 1', 'lv_s2': 'المرحلة 2',
        'acc_chart': 'الدقة لكل طريقة مع فترة ثقة 95%',
        'why': '💡 ليش ResNet50 (frozen) وليس نموذج الدمج؟ دمج الشبكات الثلاث حقق 77.8% مقابل 76.6%، لكن اختبار McNemar '
               'يبيّن أن الفرق غير دال إحصائياً (p = 0.61 بعد تصحيح Holm)، وResNet50 المجمّد نموذج واحد صغير وسريع.',
        'recall_chart': 'الـRecall لكل نوع تآكل في المرحلة 2', 'method': 'الطريقة',
        'figures': '🖼️ أشكال التقرير', 'tables': 'جداول النتائج كاملة',
        'about_md': '''
### المشكلة
الأنابيب وهياكل السفن والمنصات البحرية تتآكل في ماء البحر. الفحص اليوم يعتمد على الغواصين ومشغّلي الفيديو،
وهذا بطيء ومكلف وخطر، والحكم يعتمد على رأي الفاحص.

### الفكرة: مرحلتان
1. **المرحلة 1 - الكشف.** هل الهيكل *سليم* أم *متآكل*؟
2. **المرحلة 2 - النوع.** إذا كان متآكلاً، أي نوع من الثمانية: crevice، erosion، galvanic، MIC، pitting،
   stress، under-insulation، أو uniform؟

خطأ المرحلة 1 لا تصلحه المرحلة 2، لذلك قُيّمت المرحلتان معاً أيضاً على تسع فئات.

### البيانات
Marine Corrosion Dataset من Kaggle: 9,000 صورة في تسع فئات متوازنة. حُذفت تسع صور فارغة، وقُسّمت
الـ8,991 الباقية 70/15/15 بطريقة stratified. صور التدريب عُزّزت بالدوران والقلب وتغيير الإضاءة.

### الطرق المقارنة
HOG + SVM، GLCM + Random Forest، امتداد اللون والملمس، شبكات CNN من الصفر، وTransfer Learning بـResNet50.
كل الطرق اختُبرت على نفس الصور وقورنت باختبار McNemar.

### الحدود
كثير من صور الداتا لا تبدو صوراً حقيقية تحت الماء، فقد تكون الدقة أقل على صور فحص حقيقية. المرحلة 2 تصيب
نحو 3 مرات من 4. التطبيق يصنّف الصورة كاملة ولا يحدد مكان الضرر بدقة، والخريطة الحرارية مؤشر فقط.
''',
    },
}

# name, what it is, suggested action - per corrosion type and language
TYPES = {
    'Crevice': {
        'en': ('Crevice corrosion', 'Localised attack inside narrow gaps (under gaskets, washers, lap joints) where seawater stagnates.',
               'Open and clean the joint, inspect hidden faces, reseal the gap and check neighbouring fasteners.'),
        'ar': ('تآكل الشقوق', 'هجوم موضعي داخل الفجوات الضيقة (تحت الحشوات والورد والوصلات) حيث يركد ماء البحر.',
               'افتح الوصلة ونظّفها، افحص الأسطح المخفية، أعد إحكام الفجوة، وافحص المثبتات المجاورة.')},
    'Erosion': {
        'en': ('Erosion corrosion', 'Metal loss sped up by fast or turbulent flow, often at bends, elbows and pump parts.',
               'Measure wall thickness at bends and downstream of fittings; review flow velocity and coating.'),
        'ar': ('تآكل التعرية', 'فقد معدن يتسارع بسبب التدفق السريع أو المضطرب، غالباً عند الأكواع والانحناءات وأجزاء المضخات.',
               'قِس سماكة الجدار عند الانحناءات وبعد الوصلات، وراجع سرعة التدفق والطلاء.')},
    'Galvanic': {
        'en': ('Galvanic corrosion', 'One metal corrodes faster because it touches a different, more noble metal in seawater.',
               'Identify the two metals, isolate them electrically, and check the sacrificial anodes.'),
        'ar': ('التآكل الجلفاني', 'معدن يتآكل أسرع لأنه يلامس معدناً مختلفاً أنبل منه في ماء البحر.',
               'حدّد المعدنين، اعزلهما كهربائياً، وافحص الأنودات المضحّية.')},
    'MIC': {
        'en': ('Microbiologically influenced corrosion (MIC)', 'Bacteria and biofilms create aggressive local conditions on the metal.',
               'Take a deposit sample for microbial testing, clean the surface and review biocide treatment.'),
        'ar': ('التآكل الميكروبي (MIC)', 'البكتيريا والأغشية الحيوية تخلق ظروفاً موضعية عدوانية على المعدن.',
               'خذ عينة من الرواسب للفحص الميكروبي، نظّف السطح، وراجع المعالجة بالمبيدات الحيوية.')},
    'Pitting': {
        'en': ('Pitting corrosion', 'Small, deep holes in an otherwise intact surface; dangerous because depth is hard to see.',
               'Measure pit depth (UT or pit gauge), assess remaining wall thickness, recoat or repair.'),
        'ar': ('تآكل التنقّر', 'ثقوب صغيرة وعميقة على سطح يبدو سليماً، وخطرها أن العمق يصعب رؤيته بالعين.',
               'قِس عمق النقر (بالموجات فوق الصوتية أو بمقياس النقر)، قيّم سماكة الجدار المتبقية، ثم أعد الطلاء أو أصلح.')},
    'Stress': {
        'en': ('Stress corrosion cracking', 'Cracks that grow under tensile stress in a corrosive environment.',
               'Treat as urgent: test for cracks (dye penetrant or magnetic particle) and get an engineering assessment.'),
        'ar': ('التشقق بالتآكل الإجهادي', 'شقوق تنمو تحت إجهاد الشد في بيئة أكّالة.',
               'تعامل معه كحالة عاجلة: افحص الشقوق (بالصبغة النافذة أو الجسيمات المغناطيسية) واطلب تقييماً هندسياً.')},
    'Under Insulation': {
        'en': ('Corrosion under insulation', 'Corrosion hidden under insulation or coatings where moisture has been trapped.',
               'Remove insulation around the area, inspect the bare surface and repair the moisture barrier.'),
        'ar': ('التآكل تحت العزل', 'تآكل مخفي تحت العزل أو الطلاء حيث تنحبس الرطوبة.',
               'أزل العزل حول المنطقة، افحص السطح المكشوف، وأصلح حاجز الرطوبة.')},
    'Uniform': {
        'en': ('Uniform corrosion', 'Even metal loss over a large area; the most common and most predictable form.',
               'Measure remaining thickness, estimate the corrosion rate and plan recoating.'),
        'ar': ('التآكل المنتظم', 'فقد معدن متساوٍ على مساحة واسعة، وهو الشكل الأكثر شيوعاً والأسهل توقعاً.',
               'قِس السماكة المتبقية، قدّر معدل التآكل، وخطّط لإعادة الطلاء.')},
}
