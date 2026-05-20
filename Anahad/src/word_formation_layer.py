import time
class WordFormation:
    STATIC_STABLE_FRAMES = 3
    STATIC_STABLE_TIME   = 0.2
    STATIC_COOLDOWN      = 0.4
    DYNAMIC_COOLDOWN     = 1.5
    SPACE_TIMEOUT        = 2.5

    def __init__(self):
        self.current_word = ""
        self.sentence     = ""
        self.s_previous            = ""
        self.s_stable_count        = 0
        self.s_gesture_time        = None
        self.s_last_confirmed_time = time.time()
        self.s_last_change_time    = time.time()
        self.d_last_word      = ""
        self.d_last_fire_time = 0.0
        self.d_lock           = False
        self.last_hand_time = time.time()
        self.space_locked   = True

    def update(self, static_label, dynamic_label, mode, hand_present):
        now = time.time()

        if not hand_present:
            self.last_hand_time = now
            self.space_locked   = False
            self.d_lock         = False
            self.reset_static_state()
            if self.current_word and now - self.last_hand_time > self.SPACE_TIMEOUT:
                self.flush_word()
            return self.current_word, self.sentence

        self.last_hand_time = now

        if mode == "Dynamic" and dynamic_label and len(dynamic_label) > 1:
            cooldown_passed  = now - self.d_last_fire_time > self.DYNAMIC_COOLDOWN
            last_in_sentence = self.sentence.strip().split()[-1] if self.sentence.strip() else ""

            is_duplicate = (dynamic_label == self.d_last_word and
                            dynamic_label == last_in_sentence)

            if not is_duplicate and cooldown_passed:
                if self.current_word:
                    self.flush_word()
                self.add_to_sentence(dynamic_label)
                self.d_last_word      = dynamic_label
                self.d_last_fire_time = now
                self.d_lock           = True
                self.reset_static_state()

            return self.current_word, self.sentence

        if mode == "Static" and static_label and len(static_label) == 1:
            self.d_lock = False

            if static_label != self.s_previous:
                # New letter — restart stability timer
                self.s_previous         = static_label
                self.s_stable_count     = 1
                self.s_gesture_time     = now
                self.s_last_change_time = now
                return self.current_word, self.sentence

            if now - self.s_last_change_time < self.STATIC_COOLDOWN:
                return self.current_word, self.sentence

            self.s_stable_count    += 1
            self.s_last_change_time = now

            if (self.s_stable_count        >= self.STATIC_STABLE_FRAMES and
                    self.s_gesture_time    and
                    now - self.s_gesture_time        >= self.STATIC_STABLE_TIME and
                    now - self.s_last_confirmed_time >= self.STATIC_COOLDOWN):
                self.current_word          += static_label
                self.s_last_confirmed_time  = now
                self.s_stable_count         = 0
                self.s_gesture_time         = now

        if (not self.space_locked and self.current_word and
                now - self.last_hand_time > self.SPACE_TIMEOUT):
            self.flush_word()

        return self.current_word, self.sentence

    def backspace(self):
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            words = self.sentence.strip().split()
            self.sentence = (" ".join(words[:-1]) + " ") if len(words) > 1 else ""

    def manual_space(self):
        self.flush_word()

    def reset(self):
        self.current_word     = ""
        self.sentence         = ""
        self.d_last_word      = ""
        self.d_last_fire_time = 0.0
        self.d_lock           = False
        self.space_locked     = True
        self.reset_static_state()
    def add_to_sentence(self, word):
        self.sentence = self.sentence.rstrip() + " " + word + " "
    def flush_word(self):
        if self.current_word:
            self.add_to_sentence(self.current_word)
            self.current_word = ""
            self.reset_static_state()
            self.space_locked = True

    def reset_static_state(self):
        self.s_previous     = ""
        self.s_stable_count = 0
        self.s_gesture_time = None