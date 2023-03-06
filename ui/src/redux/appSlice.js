import { createSlice } from '@reduxjs/toolkit'

// Page width
export const widthSlice = createSlice({
  name: 'width',
  initialState: { value: 500 },
  reducers: {
    setWidth: (state, action) => {
      state.value = action.payload;
    },
  },
});
export const { setWidth } = widthSlice.actions
export const widthReducer = widthSlice.reducer

var storedTheme = localStorage.getItem('theme') || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
var darkMode = storedTheme === "dark"
const htmlEl = document.querySelector('html');
darkMode ? htmlEl.classList.add('dark') : htmlEl.classList.remove('dark');

// Model to display
export const ModelSlice = createSlice({
  name: 'Model',
  initialState: { value: '' },
  reducers: {
    setModel: (state, action) => {
      state.value = action.payload;
    },
  },
});
export const { setModel } = ModelSlice.actions
export const modelReducer = ModelSlice.reducer

// Misc
export const ViewSlice = createSlice({
  name: 'View',
  initialState: { summarizeResources: true, colorize: true, darkMode: darkMode, resourceFilter: 'All', showOptions: false, showWorkloadsTable: false, workloadOptions: [] },
  reducers: {
    setSummarizeResources: (state, action) => {
      state.summarizeResources = action.payload;
    },
    setColorize: (state, action) => {
      state.colorize = action.payload;
    },
    setDarkMode: (state, action) => {
      state.darkMode = action.payload;
    },
    setResourceFilter: (state, action) => {
      state.resourceFilter = action.payload;
    },
    setShowOptions: (state, action) => {
      state.showOptions = action.payload;
    },
    setShowWorkloadsTable: (state, action) => {
      state.showWorkloadsTable = action.payload;
    },
    addWorkloadOption: (state, action) => {
      if (!state.workloadOptions.some(obj => obj.id === action.payload.id)) {
        // If it doesn't exist, add it to the list. We don't want duplicates in the workload table.
        state.workloadOptions.push(action.payload);
      }
    },
  },
});
export const { setSummarizeResources, setColorize, setDarkMode, setResourceFilter, setShowOptions, setShowWorkloadsTable, addWorkloadOption } = ViewSlice.actions
export const viewReducer = ViewSlice.reducer

// Most recently selected
export const SelectSlice = createSlice({
  name: 'Select',
  initialState: { id: '' },
  reducers: {
    selectClick: (state, action) => {
      if (action.payload) {
        state.value = {...action.payload.op, resourceName: action.payload.resourceName};
      } else {
        state.value = null;
      }
    },
  },
});
export const { selectClick } = SelectSlice.actions
export const selectReducer = SelectSlice.reducer
