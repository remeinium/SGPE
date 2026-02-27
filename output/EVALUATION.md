================================================================================
BATTERY 1: SINHALA LINGUISTIC COMPLEXITY (2,000 Edge-Case Words)
================================================================================

  Category                        Total   Pass   Fail
  ------------------------------------------------------
  aadhyaathmika                       1      1      0
  aakhyaanaya                         1      1      0
  aathmaya                            1      1      0
  abhidhamma                          1      1      0
  adhyaapanaya                        1      1      0
  adhyaksha                           1      1      0
  aitihaasika                         1      1      0
  aniccataava                         1      1      0
  antahpuraya                         1      1      0
  antharjaathika                      1      1      0
  ashvayaa                            1      1      0
  aushadhaya                          1      1      0
  bare_hal_zwj                        1      1      0
  bare_virama                         1      1      0
  bare_zwj                            1      1      0
  braahmana                           1      1      0
  brahmaya                            1      1      0
  chandrikaa                          1      1      0
  chhandas                            1      1      0
  conjunct_anusvara                  28     28      0
  conjunct_pili_anusvara             22     22      0
  constructed_multisyllable         252    252      0
  cricket                             1      1      0
  dangling_zwj                        1      1      0
  dhammachakka                        1      1      0
  dhyaanaya                           1      1      0
  double_conjunct                    29     29      0
  dravyaya                            1      1      0
  duhkhaya                            1      1      0
  grahanaya                           1      1      0
  granthaya                           1      1      0
  indriya                             1      1      0
  jyotishya                           1      1      0
  kramaya                             1      1      0
  kshatriya                           1      1      0
  kshetraya                           1      1      0
  kshitija                            1      1      0
  mahaparinibbana                     1      1      0
  manahkalpita                        1      1      0
  mantraya                            1      1      0
  mrutyuva                            1      1      0
  multi_conjunct_sequence             1      1      0
  nibbaanaya                          1      1      0
  nirvachanaathmaka                   1      1      0
  nishkriya                           1      1      0
  paticcasamuppaada                   1      1      0
  praadeshiiyakaranaya                1      1      0
  praatibhaasika                      1      1      0
  prajaava                            1      1      0
  prakaashaya                         1      1      0
  prashast                            1      1      0
  pratipattiya                        1      1      0
  prativyuuhaathmaka                  1      1      0
  pratyaksha                          1      1      0
  pratyayaya                          1      1      0
  pratyuthpanna                       1      1      0
  praudha                             1      1      0
  premaya                             1      1      0
  quad_stack                          1      1      0
  quad_virama_chain                   1      1      0
  rakaransaya_form                    3      3      0
  ritvija                             1      1      0
  saammpradaayika                     1      1      0
  samasth                             1      1      0
  sammaasambuddha                     1      1      0
  samskrutaya                         1      1      0
  samudraya                           1      1      0
  sankhaaraya                         1      1      0
  sanskaaraya                         1      1      0
  sansthaapanaya                      1      1      0
  satyaya                             1      1      0
  saundarya                           1      1      0
  shaastraya                          1      1      0
  shaastriya                          1      1      0
  shraddhaava                         1      1      0
  shreemath                           1      1      0
  shreshtha                           1      1      0
  svaamiyaa                           1      1      0
  svabhaavaya                         1      1      0
  svachchhand                         1      1      0
  tantraya                            1      1      0
  triple_conjunct                     1      1      0
  triple_conjunct_gen                64     64      0
  trividha                            1      1      0
  udghoshanaya                        1      1      0
  upaadaanaya                         1      1      0
  upanishad                           1      1      0
  vaichitrya                          1      1      0
  vaidya                              1      1      0
  vastraya                            1      1      0
  very_long_compound                  1      1      0
  vipassanaava                        1      1      0
  vishvaasaya                         1      1      0
  vowel_prefix_conjunct               1      1      0
  vyaakaranaya                        1      1      0
  vyaapaaraya                         1      1      0
  vyatirekaya                         1      1      0
  vyavahaarika                        1      1      0
  vyavasthaava                        1      1      0
  yansaya_form                        7      7      0
  yantraya                            1      1      0
  zwnj_middle                         1      1      0

  Result: PASS — Tested 500 complex words. Violations: 0, Leading-space violations: 0

================================================================================
BATTERY 2: GLITCHED TOKEN DETECTION (v2 Multi-Script)
================================================================================
  Total unified vocab size: 328,020 (SGPE component: 128,001)
  Zero-usage SGPE tokens: 1,394
  Near-zero (< 3) tokens: 3,163

  Result: PASS — Zero: 1394, Near-Zero: 3163, Glitched: 0

================================================================================
BATTERY 3: FRONTIER BENCHMARKING (V2 STRATIFIED)
================================================================================

1. Tokenization Anatomy (Visual Examples)

'ව්යාකරණය':
  SGPE                           ['ව්යා', 'කරණය']                              (2 tokens)
  OpenAI (o200k_base)            ['ව්', 'යා', 'ක', 'රණ', 'ය']                  (5 tokens)
  Llama 4 Scout                  ['ව්', 'යා', 'කර', 'ණය']                      (4 tokens)
  DeepSeek V3                    ['ව', '්', 'ය', 'ා', 'ක', 'ර', '�', '�', 'ය'] (9 tokens)

'ශ්‍රී ලංකාව':
  SGPE                           ['ශ්\u200dරී', ' ලංකාව']                      (2 tokens)
  OpenAI (o200k_base)            ['ශ්', '\u200dරී', ' ලංක', 'ාව']              (4 tokens)
  Llama 4 Scout                  ['ශ්', '\u200dර', 'ී', ' ල', 'ං', 'ක', 'ාව']  (7 tokens)
  DeepSeek V3                    ['�', '�', '්', '\u200d', 'ර', 'ී', ' �', '�', '�', '�', 'ක', 'ා', 'ව'] (13 tokens)

'अंतर्राष्ट्रीय':
  SGPE                           ['अंतर्राष्ट्रीय']                            (1 tokens)
  OpenAI (o200k_base)            ['अ', 'ंतर', '्र', 'ाष्ट्रीय']                (4 tokens)
  Llama 4 Scout                  ['अ', 'ंतर', '्र', 'ाष्ट्रीय']                (4 tokens)
  DeepSeek V3                    ['अ', 'ंत', 'र', '्र', 'ाष', '्ट', '्री', 'य'] (8 tokens)

'कृत्रिम बुद्धिमत्ता':
  SGPE                           ['कृत्रिम', ' बुद्धिमत्ता']                   (2 tokens)
  OpenAI (o200k_base)            ['क', 'ृ', 'त्र', 'िम', ' बुद्ध', 'िम', 'त्ता'] (7 tokens)
  Llama 4 Scout                  ['क', 'ृ', 'त्र', 'िम', ' ब', 'ुद्ध', 'िम', 'त्ता'] (8 tokens)
  DeepSeek V3                    ['क', 'ृ', 'त्र', 'िम', ' ब', 'ुद', '्ध', 'िम', 'त्त', 'ा'] (10 tokens)

Evaluating 1,499,950 sentences...

====== Sinhala Results ======
Tokenizer            |       Tokens |     TWR | Chr/Tok |  % Reduction
----------------------------------------------------------------------
SGPE                 |    6,665,177 |   1.276 |    4.83 |            -
OpenAI (o200k_base)  |   17,360,196 |   3.324 |    1.85 |        61.6%
Llama 4 Scout        |   18,157,707 |   3.476 |    1.77 |        63.3%
DeepSeek V3          |   29,152,698 |   5.581 |    1.10 |        77.1%

====== Hindi Results ======
Tokenizer            |       Tokens |     TWR | Chr/Tok |  % Reduction
----------------------------------------------------------------------
SGPE                 |   13,432,763 |   1.181 |    4.29 |            -
OpenAI (o200k_base)  |   18,394,075 |   1.617 |    3.13 |        27.0%
Llama 4 Scout        |   19,566,121 |   1.720 |    2.94 |        31.3%
DeepSeek V3          |   31,682,218 |   2.786 |    1.82 |        57.6%

====== English Results ======
Tokenizer            |       Tokens |     TWR | Chr/Tok |  % Reduction
----------------------------------------------------------------------
SGPE                 |    7,240,151 |   1.330 |    4.46 |            -
OpenAI (o200k_base)  |    7,420,527 |   1.364 |    4.35 |         2.4%
Llama 4 Scout        |    7,512,843 |   1.381 |    4.30 |         3.6%
DeepSeek V3          |    7,904,670 |   1.453 |    4.09 |         8.4%

========================= OVERALL Results =========================
Tokenizer            |       Tokens |     TWR | Chr/Tok |  % Reduction
----------------------------------------------------------------------
SGPE                 |   27,338,091 |   1.241 |    4.47 |            -
OpenAI (o200k_base)  |   43,174,798 |   1.959 |    2.83 |        36.7%
Llama 4 Scout        |   45,236,671 |   2.053 |    2.70 |        39.6%
DeepSeek V3          |   68,739,586 |   3.119 |    1.78 |        60.2%

================================================================================
BATTERY 4: ROUND-TRIP CONSISTENCY
================================================================================

  Sentences tested:               1,499,950
  Total words:                   22,190,730
  Total characters tested:      122,274,117
  Total tokens generated:        27,503,859
  Mismatches (non-UNK):                   0
  Mismatches (with UNK loss):        19,320
  Crashes:                                0

  Result: PASS — Tested 1,499,950 sentences (122,274,117 chars). Non-UNK mismatches: 0, UNK-caused losses: 19320, Crashes: 0

================================================================================
BATTERY 5: BOUNDARY & LEADING SPACE EDGE-CASES 
================================================================================
  [✓] [B01-Sinhala-leading-space   ] ' සිංහල' -> '[UNK]හල'
  [✓] [B02-Sinhala-no-leading-space] 'සිංහල' -> '[UNK]හල'
  [✓] [B03-Sinhala-trailing-punct  ] 'සිංහල.' -> '[UNK]හල.'
  [✓] [B04-Sinhala-multi-word      ] 'දරුවන් පාසලට' -> 'දරුවන් පාසලට'
  [✓] [D01-Devanagari-leading-space] ' हिंदी' -> '[UNK]दी'
  [✓] [D02-Devanagari-no-leading   ] 'नमस्ते' -> 'नमस्ते'
  [✓] [D03-Devanagari-trailing-danda] 'नमस्ते।' -> 'नमस्ते।'
  [✓] [D04-Devanagari-multi-word   ] 'भारत देश' -> 'भारत देश'
  [✓] [D05-Devanagari-anusvara     ] 'संस्कृत' -> 'संस्कृत'
  [✓] [F01-SinhalaEng              ] 'සිංහලදABC' -> '[UNK]හලදABC'
  [✓] [F02-DevanagariEng           ] 'हिंदीDEF' -> '[UNK]दीDEF'
  [✓] [F03-Sinhala-Devanagari      ] 'සිංහල हिंदी' -> '[UNK]හල[UNK]दी'
  [✓] [G01-Mixed-3-scripts         ] ' සිංහල123ABCहिंदी ' -> '[UNK]හල123ABC[UNK]दी '

  Result: PASS — Violations: 0

================================================================================
BATTERY 6: ZERO-BREAKAGE GUARANTEE
================================================================================
  Testing all C + HAL + ZWJ + C pairs...
  Testing C + HAL + C pairs (implicit conjuncts)...
  Testing C + vowel_sign (all combinations)...
  Testing C + HAL (terminal virama)...
  Testing C + anusvara / visarga...
  Testing C + pili + anusvara...
  Testing triple stacks...
  Testing conjuncts with leading space...

  Result: PASS — Ran 1,703 exhaustive breakage tests. Violations: 0

================================================================================
BATTERY 6: ZERO-BREAKAGE GUARANTEE (v2 Multi-Script)
================================================================================
  Testing Devanagari C + HAL + C pairs (implicit conjuncts)...
  Testing Devanagari C + vowel_sign...
  Testing Devanagari C + HAL (terminal virama)...
  Testing Devanagari C + anusvara / visarga / chandrabindu...
  Testing Devanagari C + vowel_sign + modifier...

  Result: PASS — Devanagari Violations: 0

================================================================================
BATTERY 7: DEVANAGARI LINGUISTIC COMPLEXITY
================================================================================

  Category                      Total   Pass   Fail
  ----------------------------------------------------
  anusvara                          1      1      0
  anusvara_prefix                   5      5      0
  complex                           2      2      0
  conjunct                          3      3      0
  conjunct_anusvara                 4      4      0
  double_conjunct                   1      1      0
  double_conjunct_gen             470    470      0
  extreme_compound                  1      1      0
  matra                             3      3      0
  sanskrit                          4      4      0
  simple                            4      4      0
  super_compound                    1      1      0
  very_complex                      1      1      0

  Result: PASS — Tested 500 Devanagari words. Violations: 0

================================================================================
BATTERY 8: CODE-SWITCHING INTEGRITY
================================================================================
  [simple_sinhala_english             ]   5 tokens | ['Hello', ',', ' ශ්\u200dරී', ' ලංකාව', '!']
  [code_sinhala                       ]   5 tokens | ['const', ' x', ' =', ' ප්\u200dරකාශය', ';']
  [devanagari_english                 ]   7 tokens | ['मेरा', ' नाम', ' है', ' और', ' I', ' love', ' Python']
  [code_sinhala_mixed                 ]   9 tokens | ['function', ' foo', '()', ' {', ' return', " '", 'ශ්\u200dරී', "';"]
  [sinhala_english_mixed              ]   8 tokens | ['ශ', '\u200d', '්', '\u200d', 'රී', ' ලංකාව', ' is', ' beautiful']
  [python_devanagari_comment          ]   7 tokens | ['print', "('", 'नमस्ते', "')", ' #', ' Say', ' Hello']
  [sinhala_english_complex            ]   8 tokens | ['ඒ', ' කියන්නේ', ',', ' G', 'PE', ' Token', 'izer', ' English']
  [python_sinhala_comment             ]  10 tokens | ['for', ' i', ' in', ' range', '(', '10', '):', ' #']
  [sql_devanagari                     ]   9 tokens | ['SELECT', ' *', ' FROM', ' users', ' WHERE', ' नाम', "='", 'राम']
  [arrow_fn_sinhala                   ]  22 tokens | ['const', ' create', '_func', ' =', ' (', 'p', '1', ',']
  [math_sinhala                       ]   6 tokens | ['123', ' +', ' ', '456', ' =', ' ෆ']

  Result: PASS — Tested 13 code-switching cases. Violations: 0, Crashes: 0

================================================================================
BATTERY 9: META-VOCAB ROUND-TRIP (SGPEMetaEncoder)
================================================================================

  Sentences:     1,499,950
  Round-trip failures: 0 (100.00% lossless)
  Avg tokens/sentence: 18.3
  UNK rate: 0.08%

  Result: PASS — Tested 1,499,950 sentences. Failures: 0, Crashes: 0, Lossless: 100.00%, UNK rate: 0.08%


████████████████████████████████████████████████████████████████████████████████
█                                                                              █
█                            SGPE - BATTLE TEST REPORT                         █
█                                                                              █
████████████████████████████████████████████████████████████████████████████████

  Test Battery                                           Status         Key Metric
  ────────────────────────────────────────────────────────────────────────────────
  Linguistic Complexity (2K Sanskrit/Pali Words)         ✓ PASS       0 violations
  Glitched Token Detection (v2)                          ✓ PASS
  Frontier Benchmarking (Stratified)                     ✓ PASS
  Round-Trip Consistency (v2)                            ✓ PASS       0 mismatches
  Boundary Edge-Cases (v2)                               ✓ PASS
  Zero-Breakage Guarantee (Extended)                     ✓ PASS       0 violations
  Zero-Breakage Guarantee (v2 Devanagari)                ✓ PASS
  Devanagari Linguistic Complexity                       ✓ PASS       0 violations
  Code-Switching Integrity                               ✓ PASS       0 violations
  Meta-Vocab Round-Trip (SGPEMetaEncoder)                ✓ PASS
  ────────────────────────────────────────────────────────────────────────────────
  TOTAL                                              P:10  F:0  W:0