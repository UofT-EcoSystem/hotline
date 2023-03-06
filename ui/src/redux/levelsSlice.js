import { createSlice } from '@reduxjs/toolkit'

export const levelsSlice = createSlice({
  name: 'levels',
  initialState: { value: {} },
  reducers: {
    displayNextLevel: (state, action) => {
      const levelIdx = action.payload.idx;
      state.value[levelIdx] = {};
      state.value[levelIdx].idx = action.payload.idx;
      state.value[levelIdx].ops = action.payload.ops;
      state.value[levelIdx].op = action.payload.op;
      state.value[levelIdx].selected = action.payload.selected;
      state.value[levelIdx].name = action.payload.name;

      // Delete levels below that should no longer appear because we changed the parent
      const max_idx = Math.max(...Object.keys(state.value).map(val => parseInt(val)));
      for (let del_idx = levelIdx + 1; del_idx <= max_idx; del_idx++) {
        state.value[del_idx] = {};
      }
    },
    // Outline slices when selected
    sliceHighlight: (state, action) => {
      const levelIdx = action.payload.idx;
      state.value[levelIdx].selected = action.payload.selected;
    }
  }
});

// Action creators are generated for each case reducer function
export const { displayNextLevel, sliceHighlight } = levelsSlice.actions

export default levelsSlice.reducer