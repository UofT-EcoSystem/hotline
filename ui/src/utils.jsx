
export function checkLabelsOnTop(resourceName) {
  return resourceName.toLowerCase().includes('cpu');
}

export function checkIsOpSelected(op, selectedOp) {
  var isSelected = false;
  if (selectedOp) {
    isSelected = op.id === selectedOp.id;
  }
  return isSelected;
}

export function checkIsStartAdjusted(props) {
  // return props.levelIdx > 1 && props.resourceName.toLowerCase().includes('gpu');
  return props.resourceName.toLowerCase().includes('gpu');
}

export function checkIsSearched(searchedVal, includes, exact) {
  if ( searchedVal === false || searchedVal === null || typeof searchedVal === 'undefined' || searchedVal === '' ) {
    return true;
  }
  const searchedVal_ = searchedVal.toString().toLowerCase();

  if (includes) {
    for (const include of includes) {
      if (include && include.toString().toLowerCase().includes(searchedVal_)) {
        return true;
      }
    }
  }

  if (exact) {
    for (const exact_ of exact) {
      if (exact_ && exact_.toString().toLowerCase() === searchedVal_) {
        return true;
      }
    }
  }

  return false;
}

export function getSubSlices(op, propertyOnly) {
  var allSlices = [];
  for (const [resName, res] of Object.entries(op.resources)) {
    const slices = res.slices;
    if (!slices) {
      continue;
    }
    if (propertyOnly) {
      for (const slice of slices) {
        allSlices.push(slice[propertyOnly]);
      }
    } else {
      allSlices.push(slices)
    }
  }
  return allSlices
}