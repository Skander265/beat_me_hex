from math import inf
import random
import time
import heapq
from hex_engine.player import Player as PlayerHex
from hex_engine.action import Action
from hex_engine.state import GameState
import time

# EVALUATION CONFIGURATION CLASS
class EvalWeights:
    
    PATH_WEIGHT = 1000.0
    # rewards the distance difference in the path race

    TEMPO_BONUS = 200.0
    # bonus if it's our turn in a tight position

    DUAL_PURPOSE_CONTROL = 600.0
    # large bonus if we control a critical cell that belongs to both shortest paths

    DUAL_PURPOSE_THREAT = 40.0
    # threat or pressure on a critical empty cell

    PATH_VULNERABILITY = 50.0
    # measures how vulnerable each path is

    CENTER_BONUS = 45.0
    # small structural bonus: slight preference for the center in early game




class MyPlayer(PlayerHex):
    """        
    ALGORITHM:
    - Minimax with alpha-beta pruning
    - Iterative deepening 
    - Transposition table with Zobrist hashing
    - Beam search for move pruning (explore top K moves)
    """
    
    class TimeoutException(Exception):
        """Raised when search time limit is exceeded."""
        pass

    def __init__(self, piece_type: str, name: str = "MyPlayer", debugging: int = 0):
        """
        Args:
            piece_type: "R" (top to bottom) or "B" (left to right)
            name: Player name
            debugging: 0=silent, 1=basic stats, 2=detailed moves
        """
        super().__init__(piece_type, name)
        self.board_size = 11
        self.debugging = debugging
        
        # Search parameters 
        self.ROOT_BEAM = 22          
        self.INTERNAL_BEAM = 11
        self.MAX_DEPTH = 3
        self.MAX_TIME = 30
        
        # Transposition table
        self._tt = {}
        self._tt_limit = 100_000     
        self._init_zobrist()
        
        # Statistics (reset each turn)
        self.nodes_visited = 0
        self.prune_count = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0

    # ═══════════════════════════════════════════════════════════════════════
    # ZOBRIST HASHING (for transposition table)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _init_zobrist(self):
        """
        Initialize random bitstrings for Zobrist hashing.
        Each (row, col, piece_type) gets a unique 64-bit random number.
        """
        self._zobrist = {}
        random.seed(0xC0FFEE)  
        for i in range(self.board_size):
            for j in range(self.board_size):
                for piece_type in ["R", "B"]:
                    self._zobrist[(i, j, piece_type)] = random.getrandbits(64)

    def _board_hash(self, state: GameState) -> int:
        """
        Compute Zobrist hash of current board position.
        XOR together the random values for all pieces on the board.
        """
        env = state.get_rep().get_env()
        h = 0
        for pos, piece in env.items():
            h ^= self._zobrist[(*pos, piece.get_type())]
        return h

    def _tt_key(self, state: GameState) -> tuple:
        """
        Transposition table key: (board_hash, next_player).
        Same position with different player to move = different key.
        """
        next_player = state.get_next_player()
        side = (next_player.get_piece_type() if hasattr(next_player, "get_piece_type") 
                else next_player.get_name())
        return (self._board_hash(state), side)

    # ═══════════════════════════════════════════════════════════════════════
    # TRANSPOSITION TABLE 
    # ═══════════════════════════════════════════════════════════════════════
    
    def _tt_probe(self, state: GameState, depth: int, alpha: float, beta: float) -> tuple:
        """
        Check transposition table for previously evaluated position.
        
        BOUND TYPES:
        - EXACT: Value is precise (searched all moves)
        - LOWER: Value is >= stored (beta cutoff, might be higher)
        - UPPER: Value is <= stored (alpha cutoff, might be lower)
        
        Returns: (value, cutoff, best_move)
        """
        entry = self._tt.get(self._tt_key(state))
        if not entry:
            return None, False, None
        
        self.tt_hits += 1
        stored_depth = entry["depth"]
        val = entry["val"]
        flag = entry["flag"]
        best_move = entry.get("best")
        
        # only trust values from equal or deeper searches
        if stored_depth < depth:
            return None, False, best_move  # use move hint only
        
        # Check if we can use this value to skip searching
        if flag == "EXACT":
            self.tt_cutoffs += 1
            return val, True, best_move
        elif flag == "LOWER" and val >= beta:
            # we know value is at least 'val', and val >= beta
            # beta cutoff
            self.tt_cutoffs += 1
            return beta, True, best_move
        elif flag == "UPPER" and val <= alpha:
            # we know value is at most 'val', and val <= alpha
            # alpha cutoff
            self.tt_cutoffs += 1
            return alpha, True, best_move
        
        # Can't use value, but use move ordering hint
        return None, False, best_move

    def _tt_store(self, state: GameState, depth: int, alpha_orig: float, 
                  beta_orig: float, value: float, best_action=None):
        """
        Store search result in transposition table.
        Determines bound type based on alpha/beta windows.
        """
        # evict random entry if table is full
        if len(self._tt) > self._tt_limit:
            try:
                self._tt.pop(next(iter(self._tt)))
            except StopIteration:
                pass
        
        key = self._tt_key(state)
        existing = self._tt.get(key)
        if existing and existing.get("depth", 0) > depth:
            return
        
        # Determine bound type
        if value <= alpha_orig:
            flag = "UPPER"  # all moves were <= alpha (fail-low)
        elif value >= beta_orig:
            flag = "LOWER"  # found move >= beta (fail-high)
        else:
            flag = "EXACT"  # value is exact (within window)
        
        entry = {"val": value, "flag": flag, "depth": depth}
        if best_action:
            entry["best"] = best_action
        
        self._tt[key] = entry

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════════════════
    
    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:

        actions = list(current_state.generate_possible_light_actions())
        if not actions:
            raise ValueError("No legal moves available")
        
        env = current_state.get_rep().get_env()
        stones_on_board = len(env)
        

        if remaining_time < 5.0:
            actions = list(current_state.generate_possible_light_actions())
            return actions[0]


        if remaining_time < 30:
            panic_budget = 1 
            try:
                actions = list(current_state.generate_possible_light_actions())
                return self._minimax_root(current_state, actions, depth=2, 
                                        time_budget=panic_budget, start_time=time.time())
            except self.TimeoutException:
                ordered = self._order_actions(current_state, actions[:50])
                return ordered[0] if ordered else actions[0]
            
        #opening book
        if stones_on_board == 0:
            safe_openings = [(6, 6), (6, 7), (7, 6), (7, 7)]
            random.shuffle(safe_openings)
            for pos in safe_openings:
                for action in actions:
                    if tuple(action.data.get("position")) == tuple(pos):
                        return action
        max_depth = 0

        if stones_on_board <= 20:
            max_depth = 4
        else:
            max_depth = 5

        time_budget = self.MAX_TIME
        start_time = time.time()
        
        # Initialize best_action with a safe default
        actions = list(current_state.generate_possible_light_actions())
        best_action = actions[0]
        
        # Track the best move from the PREVIOUS completed depth
        pv_move = None 

        try:
            for depth in range(1, max_depth + 1):
                depth_start = time.time()
                
                current_best = self._minimax_root(current_state, actions, depth, 
                                                time_budget, start_time, 
                                                prev_best_move=pv_move)
                
                best_action = current_best
                pv_move = current_best 
                
                if self.debugging >= 1:
                    elapsed = time.time() - depth_start
                    print(f"[Depth {depth}] completed in {elapsed:.2f}s")

        except self.TimeoutException:
            if self.debugging >= 1:
                print(f"Timeout! Returning best move from Depth {depth-1}")
        
        return best_action

    # ═══════════════════════════════════════════════════════════════════════
    # MINIMAX SEARCH
    # ═══════════════════════════════════════════════════════════════════════
    
    def _minimax_root(self, state: GameState, actions: list, depth: int, 
                      time_budget: float, start_time: float = None, 
                      prev_best_move: Action = None) -> Action:
        """
        Optimized root search with PV-Move ordering (The Amnesia Fix).
        """
        if start_time is None:
            start_time = time.time()
        
        our_type = self.get_piece_type()
        alpha, beta = -inf, inf
        best_val = -inf
        
        # quick win check 
        for action in actions:
            if time.time() - start_time > time_budget:
                raise self.TimeoutException()
            new_state = state.apply_action(action)
            dist, _ = self._get_shortest_path(new_state, our_type)
            if dist == 0:
                if self.debugging >= 1:
                    print(f"Instant win found at {action.data.get('position')}!")
                return action
        
        # we get candidates based on static heuristics
        candidates = self._get_best_candidates(state, actions, our_type, self.ROOT_BEAM)
        
        # If we have a best move from the previous depth (prev_best_move),
        # we MUST search it first to maximize alpha-beta pruning.
        if prev_best_move:
            # Ensure prev_best is in our candidates list
            if prev_best_move not in candidates:
                candidates.append(prev_best_move)
            
            # Move it to the front
            candidates.remove(prev_best_move)
            candidates.insert(0, prev_best_move)
            
            if self.debugging >= 1:
                print(f"   [Order] Searching PV move {prev_best_move.data.get('position')} first")

        # 4. Search Loop
        best_move = candidates[0]  # Default fallback
        eval_results = []
        
        for action in candidates:
            # Time check before expensive recursive call
            if time.time() - start_time > time_budget:
                # If we timeout mid-search, we rely on the result from the PREVIOUS completed depth
                raise self.TimeoutException()
            
            next_state = state.apply_action(action)
            val = self._minimax(next_state, depth - 1, alpha, beta, False, 
                               start_time, time_budget)
            
            eval_results.append((val, action))
            self.nodes_visited += 1
            
            if val > best_val:
                best_val = val
                best_move = action
                alpha = max(alpha, val)
            
            if beta <= alpha:
                break
        
        # debugging output
        if self.debugging >= 1:
            self._print_search_stats(depth, len(candidates), len(actions))
            self._print_top_moves(state, eval_results, depth)
        
        return best_move

    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float,
                 maximizing: bool, start_time: float, time_budget: float) -> float:
        """
        Core minimax with alpha-beta pruning.
        
        1. Transposition table lookup (skip if already evaluated)
        2. Alpha-beta pruning (skip branches that can't improve result)
        3. Move ordering (search best moves first for better pruning)
        4. Beam search (only explore top K moves)
        """
        if time.time() - start_time > time_budget:
            raise self.TimeoutException()
        
        # check transposition table
        tt_val, cutoff, best_move_hint = self._tt_probe(state, depth, alpha, beta)
        if cutoff:
            return tt_val
        
        alpha_orig, beta_orig = alpha, beta
        our_type = self.get_piece_type()
        
        # Terminal node: game is over
        if state.is_done():
            scores = state.get_scores()
            bonus = depth  # Prefer faster wins
            return (1_000_000 + bonus) if scores.get(our_type, 0) > 0 else (-1_000_000 - bonus)
        
        # Leaf node: evaluate position
        if depth == 0:
            return self._evaluate(state)
        
        # Get legal moves
        all_actions = list(state.generate_possible_light_actions())
        if not all_actions:
            return self._evaluate(state)
        
        # Prune move list using beam search
        beam_width = self.INTERNAL_BEAM if depth >= 2 else self.ROOT_BEAM
        pruned = self._prune_moves(state, all_actions, beam_width)
        
        # Order moves (best first for alpha-beta)
        ordered = self._order_actions(state, pruned)
        
        # Put TT hint first if available
        if best_move_hint and best_move_hint in ordered:
            ordered.remove(best_move_hint)
            ordered.insert(0, best_move_hint)
        
        # Search all moves
        best_action = None
        
        if maximizing:
            value = -inf
            for action in ordered:
                next_state = state.apply_action(action)
                child_val = self._minimax(next_state, depth - 1, alpha, beta, 
                                         False, start_time, time_budget)
                if child_val > value:
                    value = child_val
                    best_action = action
                alpha = max(alpha, value)
                if beta <= alpha:  # Beta cutoff
                    self.prune_count += 1
                    break
        else:
            value = inf
            for action in ordered:
                next_state = state.apply_action(action)
                child_val = self._minimax(next_state, depth - 1, alpha, beta, 
                                         True, start_time, time_budget)
                if child_val < value:
                    value = child_val
                    best_action = action
                beta = min(beta, value)
                if beta <= alpha:  # Alpha cutoff
                    self.prune_count += 1
                    break
        
        # Store in transposition table
        self._tt_store(state, depth, alpha_orig, beta_orig, value, best_action)
        return value

    # ═══════════════════════════════════════════════════════════════════════
    # MOVE ORDERING & PRUNING
    # 
    # This section handles all the fast filtering we do before running Minimax.
    # ═══════════════════════════════════════════════════════════════════════
    
    def _get_best_candidates(self, state: GameState, actions: list, 
                         player_type: str, max_count: int) -> list:
        """
        Select top K moves for exploration using path-based heuristics.
        
        PRIORITIES:
        1. Dual-purpose cells (on both shortest paths) - HIGHEST
        2. Blocking opponent's path - HIGH
        3. Advancing on our path - MEDIUM
        4. Connected to our stones - LOW
        """
        if len(actions) <= max_count:
            return actions
        
        w = EvalWeights
        board = state.get_rep()
        env = board.get_env()
        opp_type = "B" if player_type == "R" else "R"
        
        # get shortest paths
        _, our_path = self._get_shortest_path(state, player_type)
        _, opp_path = self._get_shortest_path(state, opp_type)
        our_path_set = set(our_path)
        opp_path_set = set(opp_path)
        
        def score_move(action):
            pos = action.data.get("position")
            score = 0
            
            # dual-purpose cells 
            if pos in our_path_set and pos in opp_path_set:
                score += w.PATH_WEIGHT
            elif pos in opp_path_set:
                score += w.PATH_WEIGHT * 0.60
            elif pos in our_path_set:
                score += w.PATH_WEIGHT * 0.40
            
            # connectivity to our stones
            neighbors = board.get_neighbours(pos[0], pos[1])
            friendly = sum(1 for t, _ in neighbors.values() if t == player_type)
            score += friendly * 50
            
            # distance to opponent stones (tactical engagement)
            min_dist_opp = float('inf')
            for opp_pos, piece in env.items():
                if piece.get_type() == opp_type:
                    dist = abs(pos[0] - opp_pos[0]) + abs(pos[1] - opp_pos[1])
                    min_dist_opp = min(min_dist_opp, dist)
            if min_dist_opp < float('inf'):
                score += 80 / (min_dist_opp + 1)
            
            return score
        
        return sorted(actions, key=score_move, reverse=True)[:max_count]


    def _prune_moves(self, state: GameState, actions: list, beam_width: int) -> list:
        """
        Fast global pruning based on path relevance.
        No local frontier bias -> considers entire board.
        """
        if len(actions) <= beam_width:
            return actions
        
        w = EvalWeights
        board = state.get_rep()
        our_type = self.get_piece_type()
        opp_type = "B" if our_type == "R" else "R"
        
        _, our_path = self._get_shortest_path(state, our_type)
        _, opp_path = self._get_shortest_path(state, opp_type)
        our_path_set = set(our_path)
        opp_path_set = set(opp_path)
        dual_set = our_path_set & opp_path_set
        
        def score_move(action):
            pos = action.data.get("position")
            score = 0
            
            if pos in dual_set:
                score += w.PATH_WEIGHT * 0.90
            if pos in our_path_set:
                score += w.PATH_WEIGHT * 0.50
            if pos in opp_path_set:
                score += w.PATH_WEIGHT * 0.40
            
            neighbors = board.get_neighbours(pos[0], pos[1])
            friendly = sum(1 for t, _ in neighbors.values() if t == our_type)
            score += friendly * 60
            
            return score
        
        return sorted(actions, key=score_move, reverse=True)[:beam_width]


    def _order_actions(self, state: GameState, actions: list) -> list:
        """
        Order moves for alpha-beta efficiency.
        Search most promising moves first to maximize pruning.
        """
        if not actions:
            return actions
        
        w = EvalWeights
        board = state.get_rep()
        our_type = self.get_piece_type()
        opp_type = "B" if our_type == "R" else "R"
        
        _, our_path = self._get_shortest_path(state, our_type)
        _, opp_path = self._get_shortest_path(state, opp_type)
        our_path_set = set(our_path)
        opp_path_set = set(opp_path)
        
        def priority(action):
            pos = action.data.get("position")
            score = 0
            
            if pos in our_path_set and pos in opp_path_set:
                score += w.PATH_WEIGHT
            elif pos in opp_path_set:
                score += w.PATH_WEIGHT * 0.50
            elif pos in our_path_set:
                score += w.PATH_WEIGHT * 0.30
            
            neighbors = board.get_neighbours(pos[0], pos[1])
            friendly = sum(1 for t, _ in neighbors.values() if t == our_type)
            score += friendly * 50
            
            return score
        
        return sorted(actions, key=priority, reverse=True)


    # ═══════════════════════════════════════════════════════════════════════
    # SHORTEST PATH (Dijkstra)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _get_shortest_path(self, state: GameState, player_type: str) -> tuple:
        """
        Find shortest path using Dijkstra's 
        
        COST MODEL:
        - Empty cell: cost 1 (need to place stone)
        - Our stone: cost 0 (already placed)
        - Opponent stone: blocked (infinite cost)
                
        Returns: (cost, path_cells)
        - cost: 0 = already won, inf = blocked, N = need N more stones
        - path_cells: List of (row, col) tuples in shortest path
        """
        board = state.get_rep()
        env = board.get_env()
        n = self.board_size
        opp_type = "B" if player_type == "R" else "R"
        
        # define start/goal edges
        if player_type == "B":
            starts = [(i, 0) for i in range(n)]      # left edge
            goals = set((i, n-1) for i in range(n))  # right edge
        else:
            starts = [(0, j) for j in range(n)]      # top edge
            goals = set((n-1, j) for j in range(n))  # bottom edge
        
        # Dijkstra's 
        g_score = {}
        parent = {}
        pq = []
        
        def push(cell, cost, prev):
            if cost < g_score.get(cell, float('inf')):
                g_score[cell] = cost
                parent[cell] = prev
                heapq.heappush(pq, (cost, cell))
        
        for start in starts:
            piece = env.get(start)
            if piece is None:
                push(start, 1, None)
            elif piece.get_type() == player_type:
                push(start, 0, None)
        
        while pq:
            cost, cell = heapq.heappop(pq)
            
            if cost != g_score.get(cell, float('inf')):
                continue 
            
            if cell in goals:
                path = []
                current = cell
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return cost, path
            
            for neighbor_type, neighbor_pos in board.get_neighbours(cell[0], cell[1]).values():
                if neighbor_type == opp_type:
                    continue  # Blocked
                step_cost = 0 if neighbor_type == player_type else 1
                push(neighbor_pos, cost + step_cost, cell)
        
        return float('inf'), []  #no path

    # ═══════════════════════════════════════════════════════════════════════
    # EVALUATION FUNCTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def _evaluate(self, state: GameState) -> float:
        """        
        COMPONENTS (in order of importance):
        1. Path race - Primary factor (who needs fewer stones to win)
        2. Tempo - Secondary factor (who moves next matters)
        3. Dual-purpose control - Critical cell control
        4. Path vulnerability - How secure is opponent's path
        5. Center bonus - Tiny tie-breaker
        
        Depth is king
        """
        w = EvalWeights
        our_type = self.get_piece_type()
        opp_type = "B" if our_type == "R" else "R"
        
        # Calculate shortest paths for both players
        our_cost, our_path = self._get_shortest_path(state, our_type)
        opp_cost, opp_path = self._get_shortest_path(state, opp_type)
        
        #terminal cases
        if our_cost == 0:
            return 1_000_000.0  
        if opp_cost == 0:
            return -1_000_000.0 
        if our_cost == float('inf') and opp_cost == float('inf'):
            return 0.0  
        if our_cost == float('inf'):
            return -80_000.0 
        if opp_cost == float('inf'):
            return 80_000.0 
        
        score = 0.0
        
        # ─────────────────────────────────────────────────────────────────
        # COMPONENT 1: Path Race 
        # ─────────────────────────────────────────────────────────────────
        path_diff = opp_cost - our_cost
        score += path_diff * w.PATH_WEIGHT
        
        # ─────────────────────────────────────────────────────────────────
        # COMPONENT 2: Tempo 
        # ─────────────────────────────────────────────────────────────────
        # In Hex, if both players need N stones, whoever moves first wins (dual distance)
        next_player = state.get_next_player()
        is_our_turn = (next_player.get_piece_type() if hasattr(next_player, "get_piece_type")
                      else next_player.get_name()) == our_type
        
        if our_cost == opp_cost:
            if is_our_turn:
                score += w.TEMPO_BONUS
            else:
                score -= w.TEMPO_BONUS
        elif abs(our_cost - opp_cost) == 1:
            if is_our_turn:
                score += w.TEMPO_BONUS * 0.5
            else:
                score -= w.TEMPO_BONUS * 0.5
        
        # ─────────────────────────────────────────────────────────────────
        # COMPONENT 3: Dual-Purpose Cells 
        # ─────────────────────────────────────────────────────────────────
        # Cells on BOTH shortest paths are the most critical 
        # Controlling these cells is huge
        board = state.get_rep()
        env = board.get_env()
        our_path_set = set(our_path)
        opp_path_set = set(opp_path)
        dual_cells = our_path_set & opp_path_set
        
        for cell in dual_cells:
            piece = env.get(cell)
            if piece:
                if piece.get_type() == our_type:
                    score += w.DUAL_PURPOSE_CONTROL
                else:
                    score -= w.DUAL_PURPOSE_CONTROL
            else:
                # Empty dual-purpose cell - check influence
                neighbors = board.get_neighbours(cell[0], cell[1])
                our_influence = sum(1 for t, _ in neighbors.values() if t == our_type)
                opp_influence = sum(1 for t, _ in neighbors.values() if t == opp_type)
                score += (our_influence - opp_influence) * w.DUAL_PURPOSE_THREAT
        
        # ─────────────────────────────────────────────────────────────────
        # COMPONENT 4: Path Vulnerability 
        # ─────────────────────────────────────────────────────────────────
        # How many weak points does opponent's path have?
        # (empty cells on their path that we're close to)
        vulnerability_our = 0
        vulnerability_opp = 0
        
        for cell in opp_path_set:
            if env.get(cell) is None:  # Empty cell on their path
                neighbors = board.get_neighbours(cell[0], cell[1])
                our_adj = sum(1 for t, _ in neighbors.values() if t == our_type)
                if our_adj > 0:
                    vulnerability_opp += our_adj
        
        for cell in our_path_set:
            if env.get(cell) is None: 
                neighbors = board.get_neighbours(cell[0], cell[1])
                opp_adj = sum(1 for t, _ in neighbors.values() if t == opp_type)
                if opp_adj > 0:
                    vulnerability_our += opp_adj
        
        score += (vulnerability_opp - vulnerability_our) * w.PATH_VULNERABILITY
        
        # ─────────────────────────────────────────────────────────────────
        # COMPONENT 5: Center Bonus (TIE-BREAKER)
        # ─────────────────────────────────────────────────────────────────
        # Slight preference for center in early 
        if len(env) < 20:  
            center = self.board_size / 2
            for pos, piece in env.items():
                dist = abs(pos[0] - center) + abs(pos[1] - center)
                max_dist = center * 2
                bonus = (1.0 - dist / max_dist) * w.CENTER_BONUS
                if piece.get_type() == our_type:
                    score += bonus
                else:
                    score -= bonus
        
        return score

    # ═══════════════════════════════════════════════════════════════════════
    # DEBUG OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    
    def _print_search_stats(self, depth: int, explored: int, total: int):
        """Print compact search statistics."""
        print(f"[Depth {depth}] {explored}/{total} moves | "
              f"Nodes={self.nodes_visited} | Prunes={self.prune_count} | "
              f"TT-Hits={self.tt_hits} | TT-Cutoffs={self.tt_cutoffs}")

    def _print_top_moves(self, state: GameState, eval_results: list, depth: int):
        """Print top 3 moves in compact single-line format."""
        if not eval_results:
            return
        
        top_moves = sorted(eval_results, key=lambda x: x[0], reverse=True)[:3]
        
        print(f"\n{'='*70}")
        print(f"TOP {len(top_moves)} MOVES - DEPTH {depth}")
        print(f"{'='*70}")
        
        for rank, (minimax_val, action) in enumerate(top_moves, 1):
            pos = action.data.get("position")
            next_state = state.apply_action(action)
            
            static_eval = self._evaluate(next_state)
            bonus = minimax_val - static_eval
            
            our_type = self.get_piece_type()
            opp_type = "B" if our_type == "R" else "R"
            our_cost, _ = self._get_shortest_path(next_state, our_type)
            opp_cost, _ = self._get_shortest_path(next_state, opp_type)
            
            if our_cost == 0:
                print(f"#{rank} {pos} | MM:{minimax_val:+7.1f} |WINNING MOVE!")
            elif opp_cost == 0:
                print(f"#{rank} {pos} | MM:{minimax_val:+7.1f} |They win")
            elif our_cost == float('inf'):
                print(f"#{rank} {pos} | MM:{minimax_val:+7.1f} |We're blocked")
            elif opp_cost == float('inf'):
                print(f"#{rank} {pos} | MM:{minimax_val:+7.1f} |They're blocked!")
            else:
                path_diff = opp_cost - our_cost
                print(f"#{rank} {pos} | MM:{minimax_val:+7.1f} | "
                      f"Path:{our_cost}v{opp_cost}({path_diff:+d}) | "
                      f"Static:{static_eval:+7.1f} | Lookahead:{bonus:+6.1f}")
        
        print(f"{'='*70}\n")