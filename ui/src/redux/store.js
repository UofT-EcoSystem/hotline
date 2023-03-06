import { configureStore, getDefaultMiddleware  } from '@reduxjs/toolkit'
import searchReducer from './searchSlice'
import levelsReducer from './levelsSlice'
import { widthReducer, modelReducer, viewReducer, selectReducer } from './appSlice'

const logger = (store) => (next) => (action) => {
  // console.log("action fired: ");
  // console.log(action);
  next(action);
};

export const store = configureStore({
  reducer: {
    search: searchReducer,
    levels: levelsReducer,
    select: selectReducer,
    width: widthReducer,
    view: viewReducer,
    model: modelReducer
  },
  middleware: (getDefaultMiddleware) => getDefaultMiddleware().concat(logger),
});
